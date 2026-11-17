"""X9 stage 3a (ICML plan): MENTOR's mentor alone, in closed loop, tracking the LAFAN1 G1 clips on NablaEngine.

At every control step, from the current simulated state at reference frame f: plan K residual actions (setpoint =
next reference frame's joint angles + scale * a, NablaMimic's action space) with Adam through the engine, minimising a
distance between the simulated states and the reference states (terminal state f + K, plus a running term), keep the
best iterate, apply its first action, shift the plan by one step (warm start), repeat. No reward, critic or policy:
the NablaMimic reward is only used to score (mineral/envs/g1_track.py, a port checked against NablaMimic's code by
check_score_port.py). iters = 0 replays the reference joint angles as setpoints (PD tracking, the trivial baseline).

Modes:
  windows  starts at reference frames sampled on all clips (no noise), `window` steps each: for tuning K, iters, weights
  full     NablaMimic's nominal protocol: every clip once from its first frame to its end, a fall forfeits the rest;
           score = full_norm_return averaged over clips (with --max_steps the clips are cut there and scored on the
           cut length: a partial score, reported as such)

Cost: simulator steps with gradient (forward with tape + backward) and forward-only steps, per world.

  python scripts/x9_nabla/rhc_track.py --csv_dir data/lafan1_g1 --mode windows --K 16 --iters 10 --out out.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mineral.envs import g1_ref
from mineral.envs.g1_mentor import Mentor, MentorConfig
from mineral.envs.g1_track import MAX_REWARD, TERMS, G1Track, W

p = argparse.ArgumentParser()
p.add_argument('--refs', default=None, help='references npz (g1_ref); else built from --csv_dir')
p.add_argument('--csv_dir', default='data/lafan1_g1')
p.add_argument('--mode', choices=('windows', 'full'), default='windows')
p.add_argument('--K', type=int, default=16)
p.add_argument('--iters', type=int, default=10)
p.add_argument('--lr', type=float, default=0.05, help='Adam step in action units')
p.add_argument('--w_run', type=float, default=0.1, help='weight of the running cost (steps before f + K)')
p.add_argument('--w_joint', type=float, default=1.0)
p.add_argument('--w_jvel', type=float, default=0.01)
p.add_argument('--w_z', type=float, default=10.0)
p.add_argument('--w_xy', type=float, default=1.0)
p.add_argument('--w_ori', type=float, default=1.0)
p.add_argument('--w_rvel', type=float, default=0.1)
p.add_argument('--w_act', type=float, default=0.0, help='weight of mean squared residual action')
p.add_argument('--w_rate', type=float, default=0.0,
               help='weight of the squared setpoint changes over the plan (from the last applied setpoint), in action units')
p.add_argument('--inv_smooth', type=int, default=0, help='moving average of a* over +-h frames of the same clip')
p.add_argument('--init', choices=('zero', 'invert'), default='zero',
               help='new plan steps start at 0 or at the partial inversion a*(f) of the reference (G1Track.invert)')
p.add_argument('--inv_iters', type=int, default=8)
p.add_argument('--inv_damping', type=float, default=1e-2)
p.add_argument('--inv_wvel', type=float, default=0.1)
p.add_argument('--window', type=int, default=250, help='windows mode: steps per window')
p.add_argument('--n_windows', type=int, default=512)
p.add_argument('--max_steps', type=int, default=0, help='full mode: cut the clips (0: whole clips)')
p.add_argument('--clips', default='', help='comma-separated clip names (default: all)')
p.add_argument('--engine_iterations', type=int, default=10, help='Newton iterations of the engine')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--log_every', type=int, default=50)
p.add_argument('--save_traj', default='', help='npz of the executed states and setpoints (full mode)')
p.add_argument('--out', default='results/rhc_track.json')
p.add_argument('--dry_run', action='store_true', help='toy differentiable step instead of the engine (tests this script '
               'on any device; the numbers mean nothing)')
p.add_argument('--device', default='cuda')
args = p.parse_args()

torch.manual_seed(args.seed)
dev = torch.device(args.device)
t_start = time.time()
if args.refs and Path(args.refs).exists():
    refs = dict(np.load(args.refs))
else:
    names = [n for n in args.clips.split(',') if n] or None
    refs = g1_ref.build(args.csv_dir, names)
    if args.refs:
        Path(args.refs).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.refs, **refs)
if args.clips:
    keep = [str(n) for n in refs['names']].index
    sel = [keep(n) for n in args.clips.split(',') if n]
    if len(sel) != len(refs['names']):  # cut the library to the selected clips
        parts = {k: [] for k in g1_ref.KEYS}
        starts, last, off = [], [], 0
        for i in sel:
            s, e = int(refs['starts'][i]), int(refs['last'][refs['starts'][i]]) + 1
            for k in g1_ref.KEYS:
                parts[k].append(refs[k][s:e])
            starts.append(off)
            last.append(np.full(e - s, off + e - s - 1))
            off += e - s
        refs = {**{k: np.concatenate(v) for k, v in parts.items()}, 'starts': np.asarray(starts),
                'last': np.concatenate(last), 'names': refs['names'][sel], 'bodies': refs['bodies']}
t_refs = time.time() - t_start
env = G1Track(refs, device=dev, iterations=args.engine_iterations, engine=not args.dry_run)
gpu = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'
if args.dry_run:
    def toy_step(q, v, w, ctrl, kinematics=False):
        v2 = torch.cat([v[:, :6] * 0.99, 0.5 * v[:, 6:] + 10.0 * (ctrl - q[:, 7:])], 1)
        q2 = torch.cat([q[:, :3] + 0.02 * v2[:, :3], q[:, 3:7], q[:, 7:] + 0.02 * v2[:, 6:]], 1)
        if not kinematics:
            return q2, v2, w
        nb = env.m.nbody
        quat = torch.zeros(q.shape[0], nb, 4, device=q.device)
        quat[..., 0] = 1.0
        return q2, v2, w, q2, torch.zeros(q.shape[0], nb, 3, device=q.device), quat, \
            torch.zeros(q.shape[0], nb, 6, device=q.device), torch.zeros(q.shape[0], nb, 3, device=q.device)
    env.step = toy_step
    env.invert = lambda frames, **kw: (torch.zeros(frames.shape[0], env.nu, device=dev),
                                       torch.ones(frames.shape[0], device=dev))
print(f'{len(env.names)} clips, {env.last.shape[0]} frames, references in {t_refs:.0f} s, {gpu}'
      f'{" (DRY RUN: toy physics)" if args.dry_run else ""}', flush=True)

# ---------------------------------------------------------------- starts
if args.mode == 'windows':
    f_all = torch.arange(env.last.shape[0], device=dev)
    valid = f_all[f_all + args.window <= env.last]
    g = torch.Generator(device='cpu').manual_seed(args.seed)
    f0 = valid[torch.randperm(valid.shape[0], generator=g)[:args.n_windows].to(dev)]
    horizon = torch.full_like(f0, args.window)
    steps = args.window
else:
    f0 = env.starts.clone()
    horizon = env.lengths - 1
    if args.max_steps:
        horizon = torch.clamp(horizon, max=args.max_steps)
    steps = int(horizon.max())
B = f0.shape[0]
clip_of = torch.searchsorted(env.starts, f0, right=True) - 1

# ---------------------------------------------------------------- partial inversion of the visited reference frames
a_star = torch.zeros(env.last.shape[0], env.nu, device=dev)
inv = None
if args.init == 'invert':
    span = torch.arange(steps + args.K + 1, device=dev)
    need = torch.unique(torch.minimum(f0[:, None] + span[None], env.last[f0][:, None]).reshape(-1))
    t_inv = time.time()
    a_n, rel = env.invert(need, iters=args.inv_iters, damping=args.inv_damping, w_vel=args.inv_wvel)
    a_star[need] = a_n
    if args.inv_smooth:
        done = torch.zeros(env.last.shape[0], device=dev)
        done[need] = 1.0
        first = env.starts[torch.searchsorted(env.starts, need, right=True) - 1]
        acc, cnt = torch.zeros_like(a_n), torch.zeros(need.shape[0], device=dev)
        for o in range(-args.inv_smooth, args.inv_smooth + 1):
            j = torch.minimum(torch.clamp(need + o, min=0), env.last[need]).clamp(min=first)
            acc += a_star[j] * done[j, None]
            cnt += done[j]
        a_star[need] = acc / cnt[:, None]
    qs = torch.quantile(rel.float(), torch.tensor([0.1, 0.5, 0.9], device=dev)).tolist()
    inv = {'frames': int(need.shape[0]), 'seconds': time.time() - t_inv, 'rel_residual_p10_p50_p90': qs,
           'action_abs_mean': a_n.abs().mean().item(), 'action_abs_p99': torch.quantile(a_n.abs().flatten()[:10 ** 6], 0.99).item(),
           'sim_steps_fwd_with_tape': args.inv_iters * int(need.shape[0]),
           'sim_steps_fwd': (args.inv_iters + 1) * int(need.shape[0]),
           'backward_replays': args.inv_iters * 2 * env.nu * int(need.shape[0])}
    print(f'inversion of {inv["frames"]} reference transitions in {inv["seconds"]:.0f} s: relative residual '
          f'p10/p50/p90 {qs[0]:.3f}/{qs[1]:.3f}/{qs[2]:.3f}, mean |a*| {inv["action_abs_mean"]:.2f}', flush=True)

# ---------------------------------------------------------------- planner
mentor = Mentor(env, MentorConfig.from_args(args))


def plan(q, v, w, f, delta, u_prev):
    best, cost, _ = mentor.plan(q, v, w, f, delta, u_prev)
    return best, cost


# ---------------------------------------------------------------- closed loop
q, v = env.ref_state(f0)
w = torch.zeros_like(v)
f = f0.clone()
u_prev = env.q0.expand(B, -1).clone()
delta = a_star[env.next_frame(f0[:, None], torch.arange(args.K, device=dev)[None])].clone()
alive = torch.ones(B, dtype=torch.bool, device=dev)
fall_t = torch.full((B,), -1, dtype=torch.long, device=dev)
cause = torch.zeros(B, dtype=torch.long, device=dev)
tsum = torch.zeros(B, len(TERMS), device=dev)
plan_c = torch.zeros(B, device=dev)
n_fwdbwd = n_fwd = 0
traj = {'qpos': [], 'qvel': [], 'ctrl': [], 'frame': [], 'alive': []} if args.save_traj else None
t0 = time.time()
for t in range(steps):
    if args.iters:
        delta, pc = plan(q, v, w, f, delta, u_prev)
        n_fwdbwd += args.iters * args.K * B
        n_fwd += 2 * args.K * B
        plan_c += torch.where(alive, pc, torch.zeros_like(pc))
    with torch.no_grad():
        u = env.ref_qpos[env.next_frame(f), 7:] + env.action_scale * delta[:, 0]
        ctrl = torch.clamp(u, env.ctrl_lo, env.ctrl_hi)
        q, v, w, qn, xpos, xquat, cvel, scom = env.step(q, v, w, ctrl, kinematics=True)
        n_fwd += B
        f = env.next_frame(f)
        terms = env.reward_terms((qn, xpos, xquat, cvel, scom), f, u, u_prev)
        tsum += torch.where(alive[:, None], terms, torch.zeros_like(terms))
        fell, why = env.terminated(qn, f)
        new = alive & fell
        fall_t = torch.where(new, torch.full_like(fall_t, t + 1), fall_t)
        cause = torch.where(new, why, cause)
        alive = alive & ~fell & ~env.clip_ended(f) & (t + 1 < horizon)
        u_prev = u
        dead = ~alive  # finished worlds restart from their reference frame (keeps them finite; not scored)
        if dead.any():
            q[dead], v[dead], w[dead] = env.ref_qpos[f[dead]], env.ref_qvel[f[dead]], 0.0
        if traj is not None:
            for k, x in (('qpos', qn), ('qvel', v), ('ctrl', ctrl), ('frame', f), ('alive', alive)):
                traj[k].append(x.cpu())
        delta = torch.cat([delta[:, 1:], a_star[env.next_frame(f, args.K - 1)][:, None]], 1)
    if (t + 1) % args.log_every == 0 or t + 1 == steps:
        el = time.time() - t0
        ret = (tsum.sum(1) / (horizon * MAX_REWARD)).mean().item()
        print(f't {t + 1}/{steps}  alive {int(alive.sum())}/{B}  fell {int((fall_t > 0).sum())}  return so far {ret:.3f}  '
              f'{el:.0f} s  ({el / (t + 1) * 1e3:.1f} ms per control step)', flush=True)
    if not alive.any():
        break
wall = time.time() - t0

# ---------------------------------------------------------------- summary (as nablamimic.evaluate.summarize)
H = horizon.float()
nal = torch.where(fall_t > 0, fall_t.float(), H)
ret = tsum.sum(1) / (H * MAX_REWARD)
tmax = torch.tensor([abs(W[n]) * 0.02 for n in TERMS], device=dev)
out = {
    'config': vars(args),
    'gpu': gpu,
    'dry_run': args.dry_run,
    'clips': env.names,
    'worlds': B,
    'steps': steps,
    'norm_return': ret.mean().item(),
    'alive_frac': (nal / H).mean().item(),
    'failed_frac': (fall_t > 0).float().mean().item(),
    'fall_cause_counts': {c: int(((fall_t > 0) & (cause == k)).sum()) for c, k in
                          (('height', 1), ('tilt', 2), ('both', 3), ('nonfinite', 0))},
    'term_frac_alive': {n: (tsum[:, i].sum() / nal.sum() / tmax[i]).item() for i, n in enumerate(TERMS)},
    'return_per_alive_step': (tsum.sum() / nal.sum() / MAX_REWARD).item(),
    'mean_plan_cost': (plan_c.sum() / nal.sum()).item() if args.iters else None,
    'cost': {'sim_steps_fwdbwd': n_fwdbwd, 'sim_steps_fwd': n_fwd,
             'per_control_step': {'fwdbwd': args.iters * args.K, 'fwd': (2 * args.K if args.iters else 0) + 1}},
    'inversion': inv,
    'wall_s': wall,
    'ms_per_control_step': wall / max(1, t + 1) * 1e3,
    'peak_mem_gb': torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == 'cuda' else None,
}
per_clip = {}
for i in range(B):
    per_clip.setdefault(env.names[int(clip_of[i])], []).append(
        (ret[i].item(), (nal[i] / H[i]).item(), bool(fall_t[i] > 0)))
out['per_clip'] = {n: {'return': float(np.mean([r for r, _, _ in x])), 'alive_frac': float(np.mean([a for _, a, _ in x])),
                       'fell_frac': float(np.mean([fl for _, _, fl in x])), 'n': len(x)} for n, x in per_clip.items()}
if args.mode == 'full':
    out['full_norm_return' if not args.max_steps else 'partial_norm_return'] = out['norm_return']
print(json.dumps({k: out[k] for k in ('norm_return', 'alive_frac', 'failed_frac', 'fall_cause_counts',
                                      'return_per_alive_step', 'ms_per_control_step', 'peak_mem_gb')}), flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1))
if traj is not None:
    np.savez_compressed(args.save_traj, **{k: torch.stack(x).numpy() for k, x in traj.items()},
                        f0=f0.cpu().numpy(), clips=np.asarray(env.names), clip_of=clip_of.cpu().numpy())
print('wrote', args.out, flush=True)
