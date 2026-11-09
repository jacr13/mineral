"""Roll out a saved checkpoint at a target interaction budget.

The evaluation rollouts a run leaves in `<logdir>/demos/` were written when
training *stopped*, which for the reported runs was far past the budget the
paper reports (Hopper runs continued to 15-37M against a 10M budget, and one
FOCUS-OT-L2 seed has collapsed to return 180 in its rollout file while that
same seed contributes to a 0.979 mean at 10M). Fidelity computed from those
files would describe different policies than the paper's numbers.

This picks the checkpoint nearest a target step and re-rolls it, writing the
same per-episode format `plotter/fidelity_metrics.py` already consumes.

Needs an environment where dflex imports (the project's py-310 env).

    python plotter/rollout_at_budget.py --env hopper --steps 10000000 \
        --logdir plotter/workdir/.../sweep_3 --episodes 32
"""

import argparse
import importlib.util
import re
from pathlib import Path

import torch

PLOTTER_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_rr", PLOTTER_DIR / "render_rollout.py")
_rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr)


def checkpoint_nearest(ckpt_dir, target_steps):
    """Checkpoint whose filename step count is closest to `target_steps`."""
    best = None
    for p in Path(ckpt_dir).glob("*.pth"):
        m = re.search(r"steps(\d+)k", p.name)
        if not m:
            continue
        steps = int(m.group(1)) * 1000
        d = abs(steps - target_steps)
        if best is None or d < best[0]:
            best = (d, steps, p)
    if best is None:
        raise FileNotFoundError(f"no step-tagged checkpoints under {ckpt_dir}")
    return best[2], best[1]


def _make_training_env(cfg, num_envs, device):
    """The run's own env, with only the parallel-env count changed."""
    from omegaconf import OmegaConf, open_dict

    from mineral.envs.dflex import make_envs

    # plotter/utils.py shadows dflex's own top-level `utils` module, which
    # dflex/envs/ant.py imports as `from utils import load_utils`.
    _rr._prepare_dflex_path()

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        cfg.task.env.numEnvs = num_envs
        cfg.sim_device = device
    return make_envs(cfg)


def _obs_of(x):
    """dflex returns (obs, extras) from reset and 4-tuples from step."""
    return x[0] if isinstance(x, tuple) else x


def rollout(env_name, logdir, target_steps, episodes, device="cuda:0", max_len=1000, sample=False):
    from omegaconf import OmegaConf

    from mineral.agents.diffrl.models import Actor
    from mineral.common.normalizers import RunningMeanStd

    logdir = Path(logdir)
    ckpt_path, ckpt_steps = checkpoint_nearest(logdir / "ckpt", target_steps)
    cfg = OmegaConf.load(logdir / "resolved_config.yaml")
    sd = torch.load(ckpt_path, map_location=device)
    net = OmegaConf.to_container(cfg.agent.network, resolve=True)
    if net.get("encoder") is not None:
        raise NotImplementedError("only state-input runs are supported")

    # Build the env through mineral's OWN builder from the run's resolved
    # config, not a hand-written constructor call. The hand-written one in
    # render_rollout.make_env does not reproduce the training env: it missed
    # task.env.stochastic_init (mineral/cfgs/task/DFlex.yaml sets it True for
    # every DFlex task) and no_env_offset, and rollouts taken that way had an
    # initial-state spread of 0.463 against the demonstrations' 0.056, with
    # Ant's return falling from 9427 to 3755. Using the run's own config keeps
    # the initial-state distribution identical to the one the expert
    # demonstrations were drawn from, which the fidelity metric requires.
    env = _make_training_env(cfg, episodes, device)
    actor = Actor(env.num_obs, env.num_actions, **(net.get("actor_kwargs") or {})).to(device)
    actor.load_state_dict(sd["actor"])
    actor.eval()

    obs_rms = None
    if net.get("normalize_input") and sd.get("obs_rms"):
        obs_rms = RunningMeanStd((env.num_obs,)).to(device)
        obs_rms.load_state_dict({k.split(".", 1)[1]: v for k, v in sd["obs_rms"].items()})

    obs = _obs_of(env.reset())
    traj = [[] for _ in range(episodes)]
    alive = torch.ones(episodes, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(max_len):
            o = obs_rms.normalize(obs) if obs_rms is not None else obs
            mu, _, distr = actor(o)
            act = distr.rsample() if sample else mu
            step_out = env.step(act)
            nxt, rew, done = step_out[0], step_out[1], step_out[2]
            for i in range(episodes):
                if alive[i]:
                    traj[i].append((obs[i].cpu(), act[i].cpu(), rew[i].cpu()))
            alive &= ~done.bool()
            obs = nxt
            if not alive.any():
                break

    out = {}
    for i, steps in enumerate(traj):
        if len(steps) < 2:
            continue
        o = torch.stack([s[0] for s in steps])
        a = torch.stack([s[1] for s in steps])
        r = torch.stack([s[2] for s in steps])
        out[len(out)] = {"obs": {"obs": o}, "act": a,
                         "next_obs": {"obs": torch.cat([o[1:], o[-1:]])},
                         "rew": r, "done": torch.zeros(len(steps), dtype=torch.long)}
    return out, ckpt_path, ckpt_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--steps", type=int, required=True, help="target interaction budget")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions (matches how returns were evaluated). "
                         "Default is the mean action: fidelity asks whether the "
                         "learned gait matches the expert, and exploration noise "
                         "is a confound there. One Hopper checkpoint (seed1000 at "
                         "10.035M) walks fine on mean actions (4391) but falls "
                         "reproducibly when sampled (~300), while checkpoints "
                         "200k steps either side tolerate noise -- a single "
                         "brittle parameter vector should not set a fidelity number.")
    ap.add_argument("--out", default=None, help="default: <logdir>/demos_at_budget/rollouts.pt")
    args = ap.parse_args()

    data, ckpt, steps = rollout(args.env, args.logdir, args.steps, args.episodes,
                                args.device, sample=args.stochastic)
    rets = [float(v["rew"].sum()) for v in data.values()]
    out = Path(args.out or (Path(args.logdir) / "demos_at_budget" / "rollouts.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    print(f"checkpoint {ckpt.name}  (={steps/1e6:.2f}M steps, target {args.steps/1e6:.2f}M)")
    print(f"{len(data)} episodes, mean return {sum(rets)/max(1,len(rets)):.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
