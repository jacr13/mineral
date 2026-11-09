"""Render DFlex rollouts to video, for the expert-vs-learner comparison the
reviewers asked for (reviewers zDN4 "no video of the experiment" and LMQH
"could you provide expert-learner rollout comparisons").

Two sources of motion, both rendered by the same code path so the comparison is
apples-to-apples:

* ``--source demo``   -- the recorded demonstrations. The stored observations are
  invertible to the simulator's *position* state (each DFlex env exposes
  ``observation_from_state``; we invert the joint_q part of it), and dflex runs
  forward kinematics on ``state_in.joint_q`` *before* integrating, so one step
  after writing joint_q gives the exact FK of the recorded pose -- no dynamics
  and no drift.
* ``--source policy`` -- a trained actor from a run directory, rolled out in the
  simulator (deterministically by default).

Rendering does not go through dflex's USD path, which would need a GL renderer.
Instead each collision shape's capsule is transformed into world space and drawn
as a thick segment in a side view, which is readable and fully headless.

Must run in an environment where ``dflex`` imports (the project's py-310 env).

Examples
--------
Expert demonstration, SNU Humanoid::

    python plotter/render_rollout.py --source demo --env snu_humanoid \
        --demos experts/demos/DFlex_snu_demos128_return6246_len929.pt \
        --frames 300 --out videos/snu_expert.mp4

Learner from a reported run, and the two side by side::

    python plotter/render_rollout.py --source policy --env hopper \
        --logdir plotter/workdir/DFlexHopper10M-OTIL_SHAC_20260121_034126/sweep_3 \
        --frames 300 --out videos/hopper_focus_l2.mp4

    python plotter/render_rollout.py --side-by-side --env hopper \
        --demos experts/demos/DFlex_hopper_demos128_return4812_len1000.pt \
        --logdir plotter/workdir/DFlexHopper10M-OTIL_SHAC_20260121_034126/sweep_3 \
        --frames 300 --out videos/hopper_expert_vs_focus.mp4
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- geometry ----

GEO_SPHERE, GEO_BOX, GEO_CAPSULE, GEO_MESH = 0, 1, 2, 3


def _quat_rotate(q, v):
    """Rotate v by quaternion q, both as (..., 4)/(..., 3) numpy arrays.

    Same convention as dflex.util.quat_rotate (xyzw, Shepperd form).
    """
    axis = q[..., :3]
    w = q[..., 3:4]
    return (
        v * (2.0 * w * w - 1.0)
        + np.cross(axis, v) * w * 2.0
        + axis * np.sum(axis * v, axis=-1, keepdims=True) * 2.0
    )


def _quat_mul(a, b):
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack(
        (
            aw * bx + bw * ax + ay * bz - by * az,
            aw * by + bw * ay + az * bx - bz * ax,
            aw * bz + bw * az + ax * by - bx * ay,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


class ShapeSet:
    """Static collision geometry of a dflex model, grouped per environment."""

    def __init__(self, model, num_envs):
        self.shape_body = model.shape_body.detach().cpu().numpy().astype(int)
        self.geo_type = model.shape_geo_type.detach().cpu().numpy().astype(int)
        self.geo_scale = model.shape_geo_scale.detach().cpu().numpy()
        self.xform = model.shape_transform.detach().cpu().numpy()
        self.link_count = int(model.link_count)
        self.num_envs = int(num_envs)
        self.links_per_env = self.link_count // self.num_envs
        # dflex packs every env into one model; keep only env 0's shapes.
        self.keep = np.where(self.shape_body < self.links_per_env)[0]

    def skeleton_segments(self, body_X_sc, joint_parent):
        """Link-tree segments for env 0: each body origin joined to its parent.

        The SNU Humanoid's *collision* geometry is 11 coarse capsules dominated
        by two large torso spheres (its visual model in dflex is a mesh, which
        we deliberately do not depend on), so capsule silhouettes read as blobs.
        The link tree gives a legible stick figure for any model.
        """
        pos = body_X_sc[:, :3]
        segs = []
        for i in range(self.links_per_env):
            par = int(joint_parent[i])
            if par < 0 or par >= self.links_per_env:
                continue
            segs.append((pos[par], pos[i], 0.022))
        return segs

    def muscle_polylines(self, body_X_sc, muscle_start, muscle_links, muscle_points,
                         activation, strengths):
        """World-space muscle paths with their activation level, for env 0.

        Reproduces what ``SNUHumanoidEnv.render`` sends to the USD renderer --
        each muscle is a polyline through its waypoints, transformed into world
        space by the link each waypoint is attached to, coloured by
        activation / strength (which equals the policy's action, since the env
        sets ``muscle_activation = action * muscle_strengths``). We draw it
        directly rather than via USD: the installed ``usd-core`` is
        authoring-only (no UsdImagingGL / usdrecord), so a USD stage could be
        written but not rasterised here.
        """
        out = []
        n = len(muscle_start) - 1
        for mi in range(n):
            lo, hi = int(muscle_start[mi]), int(muscle_start[mi + 1])
            if hi - lo < 2:
                continue
            pts = []
            for w in range(lo, hi):
                link = int(muscle_links[w])
                if link >= self.links_per_env:
                    break
                bp, bq = body_X_sc[link, :3], body_X_sc[link, 3:]
                pts.append(bp + _quat_rotate(bq, muscle_points[w]))
            if len(pts) >= 2:
                a = float(activation[mi]) / max(float(strengths[mi]), 1e-9)
                out.append((np.array(pts), a))
        return out

    def segments(self, body_X_sc):
        """World-space capsule segments ``[(p0, p1, radius), ...]`` for env 0."""
        out = []
        for s in self.keep:
            b = self.shape_body[s]
            bp, bq = body_X_sc[b, :3], body_X_sc[b, 3:]
            sp, sq = self.xform[s, :3], self.xform[s, 3:]
            # world transform of the shape frame
            wp = bp + _quat_rotate(bq, sp)
            wq = _quat_mul(bq, sq)

            gt, gs = self.geo_type[s], self.geo_scale[s]
            if gt == GEO_CAPSULE:
                # USD capsules are +Z; dflex rotates them +90 deg about Y to get
                # the physics convention, so the capsule axis is local +X.
                half = np.array([gs[1], 0.0, 0.0])
                p0 = wp + _quat_rotate(wq, -half)
                p1 = wp + _quat_rotate(wq, half)
                out.append((p0, p1, float(gs[0])))
            elif gt == GEO_SPHERE:
                out.append((wp, wp, float(gs[0])))
            elif gt == GEO_BOX:
                half = np.array([gs[0], 0.0, 0.0])
                out.append((wp + _quat_rotate(wq, -half), wp + _quat_rotate(wq, half), float(gs[1])))
        return out


# -------------------------------------------------------------- motion src ----


def _prepare_dflex_path():
    """dflex's env modules use legacy absolute imports (``from utils import
    load_utils``, ``from envs.dflex_env import ...``), so its own package
    directory has to come first on sys.path -- otherwise ``plotter/utils.py``
    shadows dflex's ``utils`` when this script is run from the repo.
    """
    import sys

    # dflex predates NumPy 2.0, which removed these aliases outright; its
    # dflex_env.py still calls np.Inf at import time. Restore them before the
    # import rather than patching the third-party tree.
    import numpy as _np

    for _old, _new in (("Inf", "inf"), ("NAN", "nan"), ("float_", "float64"),
                       ("bool8", "bool_")):
        if not hasattr(_np, _old) and hasattr(_np, _new):
            setattr(_np, _old, getattr(_np, _new))

    import dflex

    root = os.path.dirname(dflex.__file__)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    if root not in sys.path:
        sys.path.insert(0, root)


def make_env(env_name, num_envs, device, episode_length):
    _prepare_dflex_path()
    import dflex.envs as E

    cls = {
        "hopper": "HopperEnv",
        "ant": "AntEnv",
        "humanoid": "HumanoidEnv",
        "snu_humanoid": "SNUHumanoidEnv",
        "cartpoleswingup": "CartPoleSwingUpEnv",
        "cheetah": "CheetahEnv",
    }[env_name]
    caching = {"hopper": 16, "ant": 16, "humanoid": 48, "snu_humanoid": 8,
               "cartpoleswingup": 4, "cheetah": 16}[env_name]
    return getattr(E, cls)(
        render=False, device=device, num_envs=num_envs, episode_length=episode_length,
        no_grad=True, stochastic_init=False, MM_caching_frequency=caching,
    )


# joint_q slice <- observation slice, per env. Taken from each env's own
# set_state_act / observation_from_state in dflex; only positions are needed for
# kinematics, so the velocity rows of those methods are deliberately omitted.
#   (joint_q destination slice, observation source slice)
JOINT_Q_FROM_OBS = {
    # hopper.py:235   joint_q[1:6] = obs[0:5]
    "hopper": [((1, 6), (0, 5))],
    # ant.py:243      joint_q[1] = obs[0]; [3:7] = obs[1:5]; [7:] = obs[11:19]
    "ant": [((1, 2), (0, 1)), ((3, 7), (1, 5)), ((7, None), (11, 19))],
    # humanoid.py:299 joint_q[1:2] = obs[0:1]; [3:7] = obs[1:5]; [7:] = obs[11:32]
    "humanoid": [((1, 2), (0, 1)), ((3, 7), (1, 5)), ((7, None), (11, 32))],
    # snu_humanoid.py:360 observation_from_state: [7:] -> obs[11:33]
    "snu_humanoid": [((1, 2), (0, 1)), ((3, 7), (1, 5)), ((7, None), (11, 33))],
    # cartpole_swing_up.py:177  obs = [x, xdot, sin(th), cos(th), th_dot];
    # joint_q = [x, theta]. theta is recovered as atan2(obs[2], obs[3]) and is
    # handled specially in set_joint_q_from_obs.
    "cartpoleswingup": [((0, 1), (0, 1))],
}


def set_joint_q_from_obs(env, obs, env_name):
    """Write the position part of the simulator state from an observation.

    The whole state is zeroed first. Only some joint_q entries are determined by
    the observation (horizontal translation is not), so leaving the rest from the
    previous step lets the integrator's own divergence accumulate across frames
    -- on Ant that reached NaN and the panel rendered empty. Zeroing makes each
    frame depend only on its own observation.
    """
    jq = env.state.joint_q.view(env.num_envs, -1)
    jqd = env.state.joint_qd.view(env.num_envs, -1)
    jq.zero_()
    jqd.zero_()
    if env_name == "cartpoleswingup":
        # obs stores the pole angle as (sin, cos), so invert it rather than
        # copying a slice.
        jq[:, 0] = obs[:, 0]
        jq[:, 1] = torch.atan2(obs[:, 2], obs[:, 3])
        return
    for (d0, d1), (s0, s1) in JOINT_Q_FROM_OBS[env_name]:
        tgt_w = jq[:, slice(d0, d1)].shape[1]
        val = obs[:, slice(s0, s1)]
        n = min(tgt_w, val.shape[1])
        jq[:, d0 : d0 + n] = val[:, :n]


def _emit(shapes, env, style):
    """Return (segments, muscles) for one frame; muscles is None unless asked."""
    bx = env.state.body_X_sc.detach().cpu().numpy()
    muscles = None
    if style == "muscles":
        m = env.model
        muscles = shapes.muscle_polylines(
            bx,
            m.muscle_start.detach().cpu().numpy(),
            m.muscle_links.detach().cpu().numpy(),
            m.muscle_points.detach().cpu().numpy(),
            m.muscle_activation.detach().cpu().numpy(),
            env.muscle_strengths.detach().cpu().numpy(),
        )
        segs = shapes.skeleton_segments(bx, env.model.joint_parent.detach().cpu().numpy())
    elif style == "skeleton":
        segs = shapes.skeleton_segments(bx, env.model.joint_parent.detach().cpu().numpy())
    else:
        segs = shapes.segments(bx)
    return segs, muscles


def frames_from_demo(env, env_name, demos_path, n_frames, traj, device, style="capsule"):
    d = torch.load(demos_path, map_location=device)
    obs = d["obs"]["obs"] if isinstance(d["obs"], dict) else d["obs"]
    acts = d.get("act")
    length = int(d["lengths"][traj]) if "lengths" in d else obs.shape[1]
    n = min(n_frames, length)
    shapes = ShapeSet(env.model, env.num_envs)

    env.reset()
    zero_act = torch.zeros(env.num_envs, env.num_actions, device=device)
    out = []
    for t in range(n):
        set_joint_q_from_obs(env, obs[traj : traj + 1, t].expand(env.num_envs, -1), env_name)
        # Muscle colour is activation/strength, and the env sets
        # muscle_activation = action * strengths -- so the recorded expert
        # action gives the demo panel faithful activations too.
        act = zero_act if acts is None else acts[traj : traj + 1, t].expand(env.num_envs, -1)
        # FK runs on state_in.joint_q before integration, so after this step
        # body_X_sc is exactly the FK of the pose we just wrote.
        env.step(act)
        out.append(_emit(shapes, env, style))
    return out, shapes


def load_actor(logdir, device):
    """Build the actor from a run directory and load its newest checkpoint."""
    from omegaconf import OmegaConf

    logdir = Path(logdir)
    cfg = OmegaConf.load(logdir / "resolved_config.yaml")
    ckpt_dir = logdir / "ckpt"
    cands = list(ckpt_dir.glob("*.pth"))
    if not cands:
        raise FileNotFoundError(f"no .pth under {ckpt_dir}")
    pick = None
    for pref in ("final.pth", "latest.pth"):
        hit = [c for c in cands if c.name == pref]
        if hit:
            pick = hit[0]
            break
    if pick is None:
        best = [c for c in cands if c.name.startswith("best")]
        pick = max(best or cands, key=lambda c: c.stat().st_mtime)
    return cfg, torch.load(pick, map_location=device), pick


def frames_from_policy(env, logdir, n_frames, device, sample=True, style="capsule"):
    """Roll out a trained actor. Returns (frames, shapes, checkpoint path).

    ``sample=True`` matches how the reported numbers were produced: SHAC.eval
    calls evaluate_policy(..., sample=True), i.e. stochastic actions.
    """
    from omegaconf import OmegaConf

    from mineral.agents.diffrl.models import Actor

    cfg, sd, ckpt = load_actor(logdir, device)
    net = OmegaConf.to_container(cfg.agent.network, resolve=True)
    if net.get("encoder") is not None:
        raise NotImplementedError(
            f"{logdir} uses encoder={net['encoder']}; only state-input runs are supported"
        )
    actor = Actor(env.num_obs, env.num_actions, **(net.get("actor_kwargs") or {})).to(device)
    actor.load_state_dict(sd["actor"])
    actor.eval()

    obs_rms = None
    if net.get("normalize_input") and sd.get("obs_rms"):
        from mineral.common.normalizers import RunningMeanStd

        obs_rms = RunningMeanStd((env.num_obs,)).to(device)
        obs_rms.load_state_dict({k.split(".", 1)[1]: v for k, v in sd["obs_rms"].items()})

    tanh_clamp = bool(net.get("tanh_clamp", False))
    shapes = ShapeSet(env.model, env.num_envs)
    obs = env.reset()
    out = []
    with torch.no_grad():
        for _ in range(n_frames):
            o = obs_rms.normalize(obs) if obs_rms is not None else obs
            mu, _, distr = actor(o)
            act = distr.rsample() if sample else mu
            if tanh_clamp:
                act = torch.tanh(act)
            obs, _, _, _ = env.step(act)
            out.append(_emit(shapes, env, style))
    return out, shapes, ckpt


# ---------------------------------------------------------------- drawing ----


def render_video(frame_sets, labels, out_path, fps=30, track=True, dpi=100):
    """frame_sets: list of per-source lists of segment lists."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    n_src = len(frame_sets)
    n = min(len(f) for f in frame_sets)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    colors = ["#2B6CB0", "#479A5F", "#A0522D"]

    # One shared frame for every panel and timestep, so the two motions are
    # drawn at the same scale and a size difference on screen is a real one.
    all_y, all_r = [], []
    for fs in frame_sets:
        for fr in fs[:n]:
            for s0, s1, r in fr[0]:
                all_y += [s0[1], s1[1]]
                all_r.append(r)
    pad = (max(all_r) if all_r else 0.05) * 2.0
    y_hi = (max(all_y) + pad) if all_y else 2.0

    # Half-width from the real horizontal extent about the tracked centre, not a
    # guess from height: a wide low model (Ant) otherwise has its legs clipped.
    half_x = 0.0
    for fs in frame_sets:
        for fr in fs[:n]:
            segs, muscles = fr
            xs = [p[0] for s0, s1, _ in segs for p in (s0, s1)]
            if muscles:
                xs += [float(v) for pl, _ in muscles for v in pl[:, 0]]
            if not xs:
                continue
            cx_f = float(np.mean([p[0] for s0, s1, _ in segs for p in (s0, s1)])) if segs else 0.0
            half_x = max(half_x, max(abs(x - cx_f) for x in xs))
    half_w = max(0.6, half_x + pad)

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
    try:
        for t in range(n):
            fig, axes = plt.subplots(1, n_src, figsize=(4.4 * n_src, 4.4), dpi=dpi, squeeze=False)
            for k in range(n_src):
                ax = axes[0][k]
                segs, muscles = frame_sets[k][t]
                xs = [p[0] for s in segs for p in (s[0], s[1])]
                cx = float(np.mean(xs)) if track and xs else 0.0
                # Draw each capsule as its true 2D silhouette in data
                # coordinates -- a quad between the endpoints plus a circular
                # cap at each end. Line width in points would not scale with
                # the axes, which turns thick-limbed models into blobs.
                base = np.array(matplotlib.colors.to_rgb(colors[k % len(colors)]))
                zs = [0.5 * (s0[2] + s1[2]) for s0, s1, _ in segs] or [0.0]
                z_lo, z_hi = min(zs), max(zs)
                order = np.argsort(zs)  # far limbs first
                for rank, si in enumerate(order):
                    p0, p1, r = segs[si]
                    # shade by depth so overlapping limbs stay readable
                    f = 0.0 if z_hi - z_lo < 1e-6 else (zs[si] - z_lo) / (z_hi - z_lo)
                    col = tuple(base + (1.0 - base) * (0.45 * (1.0 - f)))
                    a0 = np.array([p0[0] - cx, p0[1]])
                    a1 = np.array([p1[0] - cx, p1[1]])
                    d = a1 - a0
                    ln = float(np.hypot(*d))
                    edge = tuple(base * 0.55)
                    # Outline every capsule: on models with thick torsos (the
                    # SNU Humanoid especially) flat fills merge into a blob and
                    # the limbs become invisible.
                    if ln > 1e-9:
                        nrm = np.array([-d[1], d[0]]) / ln * r
                        quad = np.array([a0 + nrm, a1 + nrm, a1 - nrm, a0 - nrm])
                        ax.add_patch(plt.Polygon(quad, closed=True, facecolor=col,
                                                 edgecolor="none", alpha=0.85,
                                                 zorder=2 + rank))
                    for cpt in (a0, a1):
                        ax.add_patch(plt.Circle(cpt, r, facecolor=col, edgecolor="none",
                                                alpha=0.85, zorder=2 + rank))
                    # stroke the silhouette on top of the fills
                    if ln > 1e-9:
                        ang = math.degrees(math.atan2(d[1], d[0]))
                        for cpt, th1, th2 in ((a0, ang + 90, ang + 270), (a1, ang - 90, ang + 90)):
                            ax.add_patch(matplotlib.patches.Arc(
                                cpt, 2 * r, 2 * r, theta1=th1, theta2=th2,
                                edgecolor=edge, lw=0.9, zorder=2 + rank + 0.5))
                        for sgn in (1, -1):
                            off = np.array([-d[1], d[0]]) / ln * r * sgn
                            ax.plot([a0[0] + off[0], a1[0] + off[0]],
                                    [a0[1] + off[1], a1[1] + off[1]],
                                    color=edge, lw=0.9, zorder=2 + rank + 0.5)
                    else:
                        ax.add_patch(plt.Circle(cpt, r, facecolor="none",
                                                edgecolor=edge, lw=0.9, zorder=2 + rank + 0.5))
                if muscles:
                    # One polyline per muscle, coloured by activation. Low
                    # activation stays pale so the active ones stand out --
                    # the point of the figure is which muscles are firing.
                    mz = [float(np.mean(pl[:, 2])) for pl, _ in muscles] or [0.0]
                    z_lo_m, z_hi_m = min(mz), max(mz)
                    for mi, (pl, a) in enumerate(muscles):
                        # activation / strength is the raw action in [-1, 1];
                        # dflex's own renderer feeds it straight to a colour
                        # channel, so clipping at 0 matches it. Only ~16 % of
                        # expert activations exceed 0.9 and ~56 % sit below 0.1,
                        # so the inactive end has to be genuinely pale or the
                        # figure reads as uniformly red.
                        a = float(np.clip(a, 0.0, 1.0))
                        depth = 0.0 if z_hi_m - z_lo_m < 1e-6 else (mz[mi] - z_lo_m) / (z_hi_m - z_lo_m)
                        lo_c = np.array([0.86, 0.86, 0.89])   # inactive: light grey
                        hi_c = np.array([0.79, 0.00, 0.10])   # fully active: crimson
                        col = lo_c + (hi_c - lo_c) * a
                        col = tuple(col + (1.0 - col) * 0.35 * (1.0 - depth))
                        ax.plot(pl[:, 0] - cx, pl[:, 1], color=col,
                                lw=0.5 + 2.2 * a, alpha=0.16 + 0.78 * a,
                                solid_capstyle="round", zorder=40 + int(20 * a))
                ax.fill_between([-half_w, half_w], -0.2, 0.0, color="#E8E8E8", zorder=0)
                ax.axhline(0.0, color="#9A9A9A", lw=1.2, zorder=1)
                ax.set_xlim(-half_w, half_w)
                ax.set_ylim(-0.2, y_hi)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_title(labels[k], fontsize=11)
            if any(fs[t][1] for fs in frame_sets):
                fig.text(0.01, 0.015, "muscle activation: low \u2192 high",
                         fontsize=7, color="#8A8A8A", ha="left")
            fig.text(0.99, 0.015, f"t = {t}", fontsize=8, color="#666666", ha="right")
            fig.tight_layout(rect=(0, 0.03, 1, 1))
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(buf)
            plt.close(fig)
    finally:
        writer.close()
    print(f"wrote {out_path}  ({n} frames, {n_src} panel(s))")


# ------------------------------------------------------------------- main ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True,
                    choices=["hopper", "ant", "humanoid", "snu_humanoid"])
    ap.add_argument("--source", choices=["demo", "policy"], default="demo")
    ap.add_argument("--side-by-side", action="store_true",
                    help="render demo and policy in one video")
    ap.add_argument("--demos", default=None)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--label", default=None,
                    help="panel title for the policy (defaults to the checkpoint name)")
    ap.add_argument("--deterministic", action="store_true",
                    help="use the mean action; default matches eval (stochastic)")
    ap.add_argument("--style", choices=["auto", "capsule", "skeleton", "muscles"], default="auto",
                    help="auto: capsules for hopper/ant, muscles for SNU Humanoid, "
                         "link skeleton for Humanoid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episode_length = max(args.frames + 2, 1000)
    style = args.style
    if style == "auto":
        style = {"snu_humanoid": "muscles", "humanoid": "skeleton"}.get(args.env, "capsule")
    sets, labels = [], []

    if args.side_by_side or args.source == "demo":
        if not args.demos:
            raise SystemExit("--demos is required for the demo source")
        env = make_env(args.env, 1, args.device, episode_length)
        f, _ = frames_from_demo(env, args.env, args.demos, args.frames, args.traj,
                                args.device, style=style)
        sets.append(f)
        labels.append("Expert demonstration")
        del env

    if args.side_by_side or args.source == "policy":
        if not args.logdir:
            raise SystemExit("--logdir is required for the policy source")
        env = make_env(args.env, 1, args.device, episode_length)
        f, _, ckpt = frames_from_policy(env, args.logdir, args.frames, args.device,
                                        sample=not args.deterministic, style=style)
        sets.append(f)
        labels.append(args.label or f"Learner ({Path(ckpt).name})")
        del env

    render_video(sets, labels, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
