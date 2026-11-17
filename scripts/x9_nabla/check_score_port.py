"""X9: check the PyTorch port of the NablaMimic score (mineral/envs/g1_track.py) against NablaMimic's own JAX code.

CPU only (no engine): states are reference frames with perturbations, their kinematics from MuJoCo (the MJX fields the
reward reads: xpos, xquat, cvel, subtree_com); NablaMimic's TrackingEnv is built on the engine's model (its
build_model replaced by fixtures.load_model, the same mjlab G1) with absolute actions, so the action-rate argument is
a - a_prev in both. Prints the largest differences of every reward term and of the termination test.

  python scripts/x9_nabla/check_score_port.py --nablamimic_src <NablaMimic>/src --refs refs_50hz.npz
"""
import argparse
import sys

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--nablamimic_src', required=True)
p.add_argument('--refs', required=True)
p.add_argument('--n', type=int, default=64)
p.add_argument('--clips', type=int, default=3, help='use the first clips only (keeps the JAX tables small)')
args = p.parse_args()
sys.path.insert(0, args.nablamimic_src)

import types  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
from nablamimic import env as nm_env  # noqa: E402
from nablamimic.motion import MotionLibrary  # noqa: E402

# mineral/envs/__init__ imports the DFlex/Warp 1.3 envs, absent from this check's environment: load the two modules only
for name, path in (('mineral', ''), ('mineral.envs', 'envs')):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(Path(__file__).resolve().parents[2] / 'mineral' / path)]
        sys.modules[name] = pkg
from mineral.envs import g1_track  # noqa: E402
from mineral.envs.g1_ref import BM_BODIES, load_model  # noqa: E402

jax.config.update('jax_default_matmul_precision', 'highest')
z = dict(np.load(args.refs))
nf = int(z['last'][z['starts'][args.clips - 1]]) + 1
sub = {k: z[k][:nf] for k in ('qpos', 'qvel', 'body_pos', 'body_quat', 'body_lin_vel', 'body_ang_vel', 'last')}
sub.update(starts=z['starts'][:args.clips], names=z['names'][:args.clips], bodies=z['bodies'])

m = load_model()


def build_model(cfg):
    mm = load_model()
    mm.opt.iterations = cfg.solver_iterations
    kp = mm.actuator_gainprm[:mm.nu, 0].copy()
    return mm, kp


nm_env.build_model = build_model
lib = MotionLibrary(qpos=sub['qpos'], qvel=sub['qvel'], body_pos=sub['body_pos'], last=sub['last'],
                    names=list(sub['names']), starts=sub['starts'], fps=50.0, bodies=BM_BODIES,
                    body_quat=sub['body_quat'], body_lin_vel=sub['body_lin_vel'], body_ang_vel=sub['body_ang_vel'])
env = nm_env.TrackingEnv(lib, nm_env.EnvConfig(dr=False, obs_noise=False, reset_noise=False, residual_action=False))
mine = g1_track.G1Track(sub, device='cpu', engine=False)

rng = np.random.default_rng(0)
frames = rng.integers(0, nf - 1, args.n)
frames = np.minimum(frames, sub['last'][frames])
d = mujoco.MjData(m)
fields = {k: [] for k in ('qpos', 'xpos', 'xquat', 'cvel', 'subtree_com')}
for i, f in enumerate(frames):
    q, v = sub['qpos'][f].astype(np.float64).copy(), sub['qvel'][f].astype(np.float64).copy()
    s = 0.0 if i < 4 else (1.0 if i < args.n // 2 else 3.0)  # 4 exact reference states, then small and large perturbations
    q[:3] += s * rng.uniform(-0.1, 0.1, 3)
    dq = rng.normal(size=3) * 0.2 * s
    qq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qq, dq / (np.linalg.norm(dq) + 1e-9), np.linalg.norm(dq))
    q3 = np.zeros(4)
    mujoco.mju_mulQuat(q3, qq, q[3:7])
    q[3:7] = q3
    q[7:] += s * rng.normal(size=m.nu) * 0.2
    v += s * rng.normal(size=m.nv) * 0.5
    d.qpos[:], d.qvel[:] = q, v
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    fields['qpos'].append(d.qpos.copy())
    fields['xpos'].append(d.xpos.copy())
    fields['xquat'].append(d.xquat.copy())
    fields['cvel'].append(d.cvel.copy())
    fields['subtree_com'].append(d.subtree_com.copy())
F = {k: np.asarray(v, np.float32) for k, v in fields.items()}
a = rng.normal(size=(args.n, m.nu)).astype(np.float32)
a_prev = rng.normal(size=(args.n, m.nu)).astype(np.float32)


# NablaMimic (JAX)
def one(qpos, xpos, xquat, cvel, scom, f, a, ap):
    dx = env.d0.replace(qpos=qpos, xpos=xpos, xquat=xquat, cvel=cvel, subtree_com=scom)
    st = nm_env.State(dx, f, jnp.int32(1), jnp.zeros((nm_env.HIST, nm_env.PROPRIO_DIM)), ap,
                      env.sample_xi(jax.random.PRNGKey(0)), jax.random.PRNGKey(0))
    _, terms = env.reward(st, a, ap)
    return jnp.stack([terms[n] for n in g1_track.TERMS]), env.terminated(st)


ref_terms, ref_term = jax.vmap(one)(*(jnp.asarray(F[k]) for k in ('qpos', 'xpos', 'xquat', 'cvel', 'subtree_com')),
                                    jnp.asarray(frames, jnp.int32), jnp.asarray(a), jnp.asarray(a_prev))
ref_terms, ref_term = np.asarray(ref_terms), np.asarray(ref_term)

# port (PyTorch)
T = {k: torch.as_tensor(v) for k, v in F.items()}
ft = torch.as_tensor(frames, dtype=torch.long)
sc = mine.action_scale
u = mine.q0 + sc * torch.as_tensor(a)
up = mine.q0 + sc * torch.as_tensor(a_prev)
kin = (T['qpos'], T['xpos'], T['xquat'], T['cvel'], T['subtree_com'])
my_terms = mine.reward_terms(kin, ft, u, up).numpy()
my_term = mine.terminated(T['qpos'], ft)[0].numpy()

ok = True
for i, n in enumerate(g1_track.TERMS):
    diff = np.abs(my_terms[:, i] - ref_terms[:, i])
    scale = np.abs(ref_terms[:, i]).max() + 1e-12
    print(f'{n:13s} max |diff| {diff.max():.3e}  (max |term| {scale:.3e})  mean {ref_terms[:, i].mean():+.4f}')
    ok &= diff.max() <= 1e-4 * max(scale, 1e-3)
print('termination agree:', bool((my_term == ref_term).all()), f'({int(ref_term.sum())} of {args.n} terminated)')
print('exact reference states, tracking terms / max:', (my_terms[:4, :6].sum(1) / g1_track.MAX_REWARD).round(4))
ok &= bool((my_term == ref_term).all())
print('PORT OK' if ok else 'PORT MISMATCH')
sys.exit(0 if ok else 1)
