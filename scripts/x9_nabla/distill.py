"""X9 stage 3b (ICML plan): MENTOR on the LAFAN1 G1 clips, NablaEngine: a tracker policy distilled from the mentor.

- Stage I (--stage1): partial inversion a*(f) of reference transitions (G1Track.invert: LM on the joints, exact engine
  Jacobian), cloned at the reference states (history built from the reference with a* as the previous actions).
- Rounds (data aggregation): roll the policy out from reference frames sampled over the clips (restart on fall or clip
  end), label a random subset of the visited states with the mentor (warm start: the policy's own K-step plan from
  that state; Adam through the engine towards the time-indexed reference states; label = first action of the best
  plan), add (observation, label) to a FIFO buffer, regress the policy on the buffer.
- Evaluation (not counted in the cost): the same 5 s windows as rhc_track.py (same seed, same starts) and/or the full
  NablaMimic protocol (each clip once from its first frame, deterministic policy, a fall forfeits the rest).

The NablaMimic reward is used for evaluation only. Policy: NablaMimic's encoder-decoder actor and observation
(10-frame proprioception history, next 10 reference frames, reference torso orientation in the robot torso frame),
no observation noise, no domain randomisation; residual actions.

  python scripts/x9_nabla/distill.py --csv_dir data/lafan1_g1 --stage1 --rounds 200 --out results/distill.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mineral.envs import g1_ref
from mineral.envs.g1_mentor import Actor, Mentor, MentorConfig
from mineral.envs.g1_track import ANCHOR_ORI_DIM, FUTURE, GOAL_FRAME_DIM, HIST, MAX_REWARD, PROPRIO_DIM, TERMS, G1Track

p = argparse.ArgumentParser()
p.add_argument('--refs', default=None)
p.add_argument('--csv_dir', default='data/lafan1_g1')
# mentor (same names and defaults as rhc_track.py)
p.add_argument('--K', type=int, default=16)
p.add_argument('--iters', type=int, default=10)
p.add_argument('--lr', type=float, default=0.05)
p.add_argument('--w_run', type=float, default=1.0)
p.add_argument('--w_joint', type=float, default=1.0)
p.add_argument('--w_jvel', type=float, default=0.01)
p.add_argument('--w_z', type=float, default=100.0)
p.add_argument('--w_xy', type=float, default=1.0)
p.add_argument('--w_ori', type=float, default=10.0)
p.add_argument('--w_rvel', type=float, default=1.0)
p.add_argument('--w_act', type=float, default=0.0)
p.add_argument('--w_rate', type=float, default=0.0)
# Stage I
p.add_argument('--stage1', action='store_true')
p.add_argument('--stage1_windows', type=int, default=2048, help='reference windows of stage1_len frames to invert')
p.add_argument('--stage1_len', type=int, default=64)
p.add_argument('--inv_iters', type=int, default=8)
p.add_argument('--inv_smooth', type=int, default=2)
p.add_argument('--stage1_steps', type=int, default=3000)
# rounds
p.add_argument('--rounds', type=int, default=200)
p.add_argument('--hours', type=float, default=0.0, help='wall-clock budget (0: rounds only)')
p.add_argument('--envs', type=int, default=4096)
p.add_argument('--rollout', type=int, default=32, help='policy steps per round')
p.add_argument('--labels', type=int, default=16384, help='mentor labels per round')
p.add_argument('--label_chunk', type=int, default=4096)
p.add_argument('--act_noise', type=float, default=0.05)
p.add_argument('--buffer', type=int, default=262144)
p.add_argument('--policy_lr', type=float, default=3e-4)
p.add_argument('--batch', type=int, default=4096)
p.add_argument('--sgd_steps', type=int, default=200)
p.add_argument('--obs_anchor_pos', action='store_true',
               help='add the reference torso position in the robot torso frame to the policy input (not in NablaMimic)')
p.add_argument('--grad_clip', type=float, default=1.0, help='global gradient-norm clip of the policy update')
p.add_argument('--label_max', type=float, default=10.0, help='drop labels with |a| above this (action units)')
p.add_argument('--obs_max', type=float, default=100.0, help='drop samples with |observation| above this')
p.add_argument('--vel_max', type=float, default=50.0, help='restart worlds whose |qvel| exceeds this')
# evaluation
p.add_argument('--eval_every', type=int, default=10)
p.add_argument('--eval_windows', type=int, default=512)
p.add_argument('--eval_window', type=int, default=250)
p.add_argument('--eval_full', action='store_true', help='also run the full protocol at each evaluation')
p.add_argument('--engine_iterations', type=int, default=10)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='results/distill.json')
p.add_argument('--save_policy', default='')
p.add_argument('--dry_run', action='store_true', help='toy physics on any device: tests the script only')
p.add_argument('--device', default='cuda')
args = p.parse_args()

torch.manual_seed(args.seed)
dev = torch.device(args.device)
torch.backends.cuda.matmul.allow_tf32 = True
t_start = time.time()
refs = dict(np.load(args.refs)) if args.refs and Path(args.refs).exists() else g1_ref.build(args.csv_dir)
env = G1Track(refs, device=dev, iterations=args.engine_iterations, engine=not args.dry_run)
if args.dry_run:
    class ToyFFI:
        @staticmethod
        def kinematics(q, v, com):
            n, nb = q.shape[0], env.m.nbody
            xq = torch.zeros(n, nb, 4, device=q.device)
            xq[..., 0] = 1.0
            z = torch.zeros(n, nb, 3, device=q.device)
            return q, z, xq, torch.zeros(n, nb, 6, device=q.device), z

    def toy_step(q, v, w, ctrl, kinematics=False):
        v2 = torch.cat([v[:, :6] * 0.99, 0.5 * v[:, 6:] + 10.0 * (ctrl - q[:, 7:])], 1)
        q2 = torch.cat([q[:, :3] + 0.02 * v2[:, :3], q[:, 3:7], q[:, 7:] + 0.02 * v2[:, 6:]], 1)
        return (q2, v2, w) + (ToyFFI.kinematics(q2, v2, None) if kinematics else ())

    env.ffi, env.step = ToyFFI, toy_step
    env.invert = lambda frames, **kw: (0.1 * torch.randn(frames.shape[0], env.nu, device=dev),
                                       torch.ones(frames.shape[0], device=dev))
mentor = Mentor(env, MentorConfig.from_args(args))
NF = env.last.shape[0]
f_all = torch.arange(NF, device=dev)
starts_valid = f_all[f_all < env.last]
gen = torch.Generator(device='cpu').manual_seed(args.seed + 1)
HIST_DIM, GOAL_DIM = HIST * PROPRIO_DIM, FUTURE * GOAL_FRAME_DIM
ANC_DIM = ANCHOR_ORI_DIM + (3 if args.obs_anchor_pos else 0)
policy = Actor(GOAL_DIM, HIST_DIM, ANC_DIM, env.nu).to(dev)
opt = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
cost = {'stage1_fwd': 0, 'stage1_fwd_tape': 0, 'stage1_bwd_replays': 0, 'rollout_fwd': 0, 'label_fwd': 0,
        'label_fwdbwd': 0}
GPU = torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'
print(f'{len(env.names)} clips, {NF} frames, {GPU}{" (DRY RUN)" if args.dry_run else ""}, references in '
      f'{time.time() - t_start:.0f} s', flush=True)


def sim_steps():
    """Simulator steps: every step call counts its batch size once, forward-only or with a tape.

    Backward passes and tape replays are not steps.
    """
    return sum(v for k, v in cost.items() if k != 'stage1_bwd_replays')


def com0(n):
    return torch.zeros(n, 3, device=dev)


def torso_pose(q, v):
    """(B, 7): torso position and orientation of the states."""
    k = env.ffi.kinematics(q.contiguous(), v.contiguous(), com0(q.shape[0]))
    return torch.cat([k[1][:, env.torso_id], k[2][:, env.torso_id]], -1)


def anchor_obs(tq, f):
    """The goal's anchor part: reference torso orientation (6), and with --obs_anchor_pos its position (3)."""
    ori = env.anchor_ori(tq[:, 3:], f)
    return torch.cat([ori, env.anchor_pos(tq[:, :3], tq[:, 3:], f)], -1) if args.obs_anchor_pos else ori


# ---------------------------------------------------------------- buffer
class Buffer:
    def __init__(self, n):
        self.hist = torch.zeros(n, HIST_DIM, device=dev)
        self.goal = torch.zeros(n, GOAL_DIM, device=dev)
        self.anc = torch.zeros(n, ANC_DIM, device=dev)
        self.y = torch.zeros(n, env.nu, device=dev)
        self.n, self.i, self.full = n, 0, False

    def add(self, hist, goal, anc, y):
        k = hist.shape[0]
        idx = (self.i + torch.arange(k, device=dev)) % self.n
        self.hist[idx], self.goal[idx], self.anc[idx], self.y[idx] = hist, goal, anc, y
        self.full |= self.i + k >= self.n
        self.i = (self.i + k) % self.n

    def size(self):
        return self.n if self.full else self.i

    def sample(self, b):
        idx = torch.randint(0, self.size(), (b,), device=dev)
        return self.hist[idx], self.goal[idx], self.anc[idx], self.y[idx]


buf = Buffer(args.buffer)


stats = {'skipped_updates': 0, 'dropped_labels': 0, 'vel_resets': 0}


def add_checked(hist, goal, anc, y):
    """Add samples whose observation and label are finite and within bounds (counted in stats otherwise)."""
    ok = (torch.isfinite(y).all(-1) & (y.abs() <= args.label_max).all(-1)
          & torch.isfinite(hist).all(-1) & (hist.abs() <= args.obs_max).all(-1)
          & torch.isfinite(goal).all(-1) & torch.isfinite(anc).all(-1))
    stats['dropped_labels'] += int((~ok).sum())
    if ok.any():
        buf.add(hist[ok], goal[ok], anc[ok], y[ok])


def train(steps):
    losses = []
    for _ in range(steps):
        h, g, a, y = buf.sample(args.batch)
        loss = ((policy(h, g, a) - y) ** 2).mean()
        if not torch.isfinite(loss):
            stats['skipped_updates'] += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        opt.step()
        losses.append(loss.detach())
    return torch.stack(losses).mean().item() if losses else None


# ---------------------------------------------------------------- Stage I: invert reference windows, clone a*
history = {'config': vars(args), 'gpu': GPU, 'rounds': [], 'evals': []}
if args.stage1:
    t1 = time.time()
    L = args.stage1_len + HIST
    ok = f_all[(f_all + L <= env.last)]
    w0 = ok[torch.randperm(ok.shape[0], generator=gen)[:args.stage1_windows].to(dev)]
    frames = (w0[:, None] + torch.arange(L, device=dev)[None]).reshape(-1)
    a_star, rel = env.invert(frames, iters=args.inv_iters, w_vel=0.1)
    a_star = a_star.reshape(-1, L, env.nu)
    if args.inv_smooth:
        h = args.inv_smooth
        pad = torch.cat([a_star[:, :1].expand(-1, h, -1), a_star, a_star[:, -1:].expand(-1, h, -1)], 1)
        a_star = torch.stack([pad[:, i:i + L] for i in range(2 * h + 1)]).mean(0)
    n = frames.shape[0]
    cost['stage1_fwd_tape'] += args.inv_iters * n
    cost['stage1_fwd'] += (args.inv_iters + 1) * n
    cost['stage1_bwd_replays'] += args.inv_iters * 2 * env.nu * n
    fr = frames.reshape(-1, L)
    q, v = env.ref_qpos[fr], env.ref_qvel[fr]
    prev = torch.cat([torch.zeros_like(a_star[:, :1]), a_star[:, :-1]], 1)
    prop = env.proprio(q.reshape(-1, 36), v.reshape(-1, 35), prev.reshape(-1, env.nu)).reshape(-1, L, PROPRIO_DIM)
    ident = torch.tensor([1.0, 0, 0, 0, 1.0, 0] + [0.0] * (ANC_DIM - ANCHOR_ORI_DIM), device=dev)  # reference states
    for t in range(HIST, L):
        hist = prop[:, t - HIST + 1:t + 1].reshape(-1, HIST_DIM)
        buf.add(hist, env.goal_ref(fr[:, t]), ident.expand(fr.shape[0], -1), a_star[:, t])
    loss = train(args.stage1_steps)
    s1 = {'transitions': n, 'seconds': time.time() - t1, 'rel_residual_p50': rel.median().item(),
          'samples': buf.size(), 'bc_loss': loss}
    history['stage1'] = s1
    print(f'stage I: {n} transitions inverted (residual p50 {s1["rel_residual_p50"]:.3f}), {buf.size()} samples, '
          f'BC loss {loss:.4f}, {s1["seconds"]:.0f} s', flush=True)


# ---------------------------------------------------------------- environment state for policy rollouts
class Rollout:
    """A batch of worlds driven by the policy (hist, prev action, setpoint, torso orientation tracked)."""

    def __init__(self, f):
        self.reset_all(f)

    def reset_all(self, f):
        n = f.shape[0]
        self.f = f.clone()
        self.q, self.v = env.ref_state(f)
        self.w = torch.zeros_like(self.v)
        self.prev_a = torch.zeros(n, env.nu, device=dev)
        self.u_prev = env.q0.expand(n, -1).clone()
        self.hist = env.proprio(self.q, self.v, self.prev_a)[:, None].repeat(1, HIST, 1)
        self.tq = torso_pose(self.q, self.v)

    def reset(self, idx, f):
        if idx.numel() == 0:
            return
        q, v = env.ref_state(f)
        self.f[idx], self.q[idx], self.v[idx], self.w[idx] = f, q, v, 0.0
        self.prev_a[idx], self.u_prev[idx] = 0.0, env.q0
        self.hist[idx] = env.proprio(q, v, torch.zeros(f.shape[0], env.nu, device=dev))[:, None].repeat(1, HIST, 1)
        self.tq[idx] = torso_pose(q, v)

    def obs(self):
        return self.hist.reshape(-1, HIST_DIM), env.goal_ref(self.f), anchor_obs(self.tq, self.f)

    @torch.no_grad()
    def step(self, a):
        """Apply residual action a; returns (fell, ended, reward terms) after the step."""
        u = env.ref_qpos[env.next_frame(self.f), 7:] + env.action_scale * a
        ctrl = torch.clamp(u, env.ctrl_lo, env.ctrl_hi)
        self.q, self.v, self.w, qn, xpos, xquat, cvel, scom = env.step(self.q, self.v, self.w, ctrl, kinematics=True)
        self.f = env.next_frame(self.f)
        terms = env.reward_terms((qn, xpos, xquat, cvel, scom), self.f, u, self.u_prev)
        self.u_prev, self.prev_a = u, a
        self.tq = torch.cat([xpos[:, env.torso_id], xquat[:, env.torso_id]], -1)
        self.hist = torch.cat([self.hist[:, 1:], env.proprio(qn, self.v, a)[:, None]], 1)
        fell, _ = env.terminated(qn, self.f)
        return fell, env.clip_ended(self.f), terms

    def sub(self, idx):
        r = Rollout.__new__(Rollout)
        for k in ('f', 'q', 'v', 'w', 'prev_a', 'u_prev', 'hist', 'tq'):
            setattr(r, k, getattr(self, k)[idx].clone())
        return r


def sample_starts(n):
    return starts_valid[torch.randint(0, starts_valid.shape[0], (n,), generator=gen).to(dev)]


@torch.no_grad()
def act(r, noise=0.0):
    a = policy(*r.obs())
    return a + noise * torch.randn_like(a) if noise else a


# ---------------------------------------------------------------- evaluation (not counted)
@torch.no_grad()
def evaluate(f0, horizon):
    r = Rollout(f0)
    n = f0.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fall_t = torch.full((n,), -1, dtype=torch.long, device=dev)
    tsum = torch.zeros(n, len(TERMS), device=dev)
    for t in range(int(horizon.max())):
        fell, ended, terms = r.step(act(r))
        tsum += torch.where(alive[:, None], terms, torch.zeros_like(terms))
        fall_t = torch.where(alive & fell, torch.full_like(fall_t, t + 1), fall_t)
        alive = alive & ~fell & ~ended & (t + 1 < horizon)
        if not alive.any():
            break
        dead = (~alive).nonzero()[:, 0]  # finished worlds restart from their reference frame (not scored)
        r.reset(dead, r.f[dead])
    H = horizon.float()
    return {'norm_return': (tsum.sum(1) / (H * MAX_REWARD)).mean().item(), 'failed_frac': (fall_t > 0).float().mean().item(),
            'alive_frac': (torch.where(fall_t > 0, fall_t.float(), H) / H).mean().item(),
            'action_rate_per_step': (tsum[:, TERMS.index('action_rate')].sum() /
                                     torch.where(fall_t > 0, fall_t.float(), H).sum()).item()}


valid_w = f_all[f_all + args.eval_window <= env.last]
g_eval = torch.Generator(device='cpu').manual_seed(0)  # rhc_track.py's windows (seed 0)
eval_f0 = valid_w[torch.randperm(valid_w.shape[0], generator=g_eval)[:args.eval_windows].to(dev)]


def run_eval(rnd):
    t = time.time()
    e = {'round': rnd, 'windows': evaluate(eval_f0, torch.full_like(eval_f0, args.eval_window))}
    if args.eval_full:
        e['full'] = evaluate(env.starts.clone(), env.lengths - 1)
    e['seconds'] = time.time() - t
    e['sim_steps'] = sim_steps()
    e['wall_s'] = time.time() - t_start
    history['evals'].append(e)
    print(f'EVAL round {rnd}: windows return {e["windows"]["norm_return"]:.3f} fell {e["windows"]["failed_frac"]:.3f}'
          + (f' | full return {e["full"]["norm_return"]:.3f} fell {e["full"]["failed_frac"]:.3f}' if 'full' in e else '')
          + f' | sim steps {e["sim_steps"] / 1e6:.1f} M, {e["wall_s"] / 60:.0f} min', flush=True)


# ---------------------------------------------------------------- rounds
run_eval(0)
roll = Rollout(sample_starts(args.envs))
per_step = max(1, args.labels // args.rollout)
for rnd in range(1, args.rounds + 1):
    tr = time.time()
    snaps = []
    falls = 0
    for _ in range(args.rollout):
        pick = torch.randperm(args.envs, device=dev)[:per_step]
        snaps.append(roll.sub(pick))
        a = act(roll, args.act_noise)
        fell, ended, _ = roll.step(a)
        cost['rollout_fwd'] += args.envs
        falls += int(fell.sum())
        wild = ~(roll.v.abs() <= args.vel_max).all(-1)  # exploding (or non-finite) velocities: restart
        stats['vel_resets'] += int((wild & ~fell).sum())
        done = (fell | ended | wild).nonzero()[:, 0]
        roll.reset(done, sample_starts(done.shape[0]))
    # label the snapshots with the mentor, warm-started from the policy's own K-step plan
    labelled = 0
    imp = []
    snap = snaps[0]
    for s in snaps[1:]:
        for k in ('f', 'q', 'v', 'w', 'prev_a', 'u_prev', 'hist', 'tq'):
            setattr(snap, k, torch.cat([getattr(snap, k), getattr(s, k)]))
    n_lab = snap.f.shape[0]
    for c in torch.arange(n_lab, device=dev).split(args.label_chunk):
        s = snap.sub(c)
        obs = s.obs()
        q0_, v0_, w0_, f0_, up0 = s.q.clone(), s.v.clone(), s.w.clone(), s.f.clone(), s.u_prev.clone()
        plan0 = []
        sim = s.sub(torch.arange(c.shape[0], device=dev))
        for _ in range(args.K):
            a = act(sim)
            plan0.append(a)
            sim.step(a)
        cost['label_fwd'] += args.K * c.shape[0]
        best, bc, c0 = mentor.plan(q0_, v0_, w0_, f0_, torch.stack(plan0, 1), up0)
        fb, fo = mentor.sim_steps(c.shape[0])
        cost['label_fwdbwd'] += fb
        cost['label_fwd'] += fo
        add_checked(*obs, best[:, 0])
        imp.append((bc / c0.clamp_min(1e-9)).median())
        labelled += c.shape[0]
    loss = train(args.sgd_steps)
    r = {'round': rnd, 'labels': labelled, **stats, 'falls_per_1k_steps': 1000 * falls / (args.envs * args.rollout),
         'label_cost_ratio_p50': torch.stack(imp).mean().item(), 'loss': loss, 'buffer': buf.size(),
         'sim_steps': sim_steps(), 'seconds': time.time() - tr, 'wall_s': time.time() - t_start}
    history['rounds'].append(r)
    print(f'round {rnd}: falls/1k steps {r["falls_per_1k_steps"]:.1f}, label cost / policy plan cost {r["label_cost_ratio_p50"]:.3f}, '
          f'loss {loss:.4f}, {r["seconds"]:.0f} s, sim steps {r["sim_steps"] / 1e6:.1f} M', flush=True)
    if rnd % args.eval_every == 0 or rnd == args.rounds:
        run_eval(rnd)
    if args.hours and time.time() - t_start > args.hours * 3600:
        if history['evals'][-1]['round'] != rnd:
            run_eval(rnd)
        break

history['cost'] = cost
history['sim_steps'] = sim_steps()
history['wall_s'] = time.time() - t_start
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(history, indent=1))
if args.save_policy:
    torch.save(policy.state_dict(), args.save_policy)
print('wrote', args.out, flush=True)
