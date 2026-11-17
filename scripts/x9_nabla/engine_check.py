"""X9 stage 1 (ICML plan): NablaEngine checks on a cluster GPU, before any MENTOR code uses the engine.

1. reset: stepping the same stored state twice gives bitwise the same output (MENTOR's one-step map needs it);
2. gradient: d(qpos', qvel') / d ctrl from autograd against central finite differences, on a few states;
3. tape replay: one forward with a tape, two backward passes with different cotangents, each equal to a fresh
   autograd backward (then a full step Jacobian costs one forward, which DFlex cannot do);
4. throughput: control steps per second, forward only and forward + backward, at several batch sizes.

usage (inside candidj0/mineral:nabla): python scripts/x9_nabla/engine_check.py [--out results/engine_check.json]
"""
import argparse
import json
import time

import torch
from nablaengine import fixtures, torch_ffi

p = argparse.ArgumentParser()
p.add_argument('--out', default='results/engine_check.json')
p.add_argument('--fd_states', type=int, default=8)
p.add_argument('--eps', type=float, default=1e-3)
args = p.parse_args()

dev = torch.device('cuda')
torch_ffi.init(dev)
name = fixtures.default_corpus()
z = fixtures.load_corpus(name)
N = z['qpos'].shape[0]
keys = ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'friction', 'com')
report = {'corpus': name, 'corpus_states': int(N), 'gpu': torch.cuda.get_device_name(dev)}
print(report)


def batch(n, start=0):
    idx = [(start + i) % N for i in range(n)]
    return [torch.from_numpy(z[k][idx]).to(dev).contiguous() for k in keys]


# 1. reset / determinism
qpos, qvel, warm, ctrl, fr, com = batch(256)
o1 = torch_ffi.step(qpos, qvel, warm, ctrl, fr, com)
o2 = torch_ffi.step(qpos.clone(), qvel.clone(), warm.clone(), ctrl.clone(), fr, com)
report['reset_bitwise_equal'] = bool(all(torch.equal(a, b) for a, b in zip(o1, o2, strict=False)))
print('reset bitwise equal:', report['reset_bitwise_equal'])

# 2. gradient vs central finite differences (float32: compare directions and relative errors)
n = args.fd_states
qpos, qvel, warm, ctrl, fr, com = batch(n, start=1000)
g_q = torch.randn_like(qpos)
g_v = torch.randn_like(qvel)


def objective(c):
    q1, v1, _ = torch_ffi.step(qpos, qvel, warm, c, fr, com)
    return (q1 * g_q).sum(-1) + (v1 * g_v).sum(-1)  # one scalar per state


c = ctrl.clone().requires_grad_(True)
objective(c).sum().backward()
grad_ad = c.grad.detach()
grad_fd = torch.zeros_like(ctrl)
with torch.no_grad():
    for j in range(ctrl.shape[1]):
        e = torch.zeros_like(ctrl)
        e[:, j] = args.eps
        grad_fd[:, j] = (objective(ctrl + e) - objective(ctrl - e)) / (2 * args.eps)
cos = torch.nn.functional.cosine_similarity(grad_ad, grad_fd, dim=1)
rel = (grad_ad - grad_fd).norm(dim=1) / grad_fd.norm(dim=1).clamp_min(1e-12)
report['grad_cosine_median'] = cos.median().item()
report['grad_cosine_min'] = cos.min().item()
report['grad_rel_err_median'] = rel.median().item()
print(f'gradient vs FD: cosine median {cos.median():.4f} min {cos.min():.4f}, rel err median {rel.median():.3e}')

# 3. tape replay: two cotangents from one forward
qpos, qvel, warm, ctrl, fr, com = batch(256, start=2000)
out = torch_ffi.forward(qpos, qvel, warm, ctrl, fr, com, tape=True)
tape = out[4]
gq = [torch.randn_like(qpos) for _ in range(2)]
gv = [torch.randn_like(qvel) for _ in range(2)]
ok = []
for k in range(2):
    g_low = torch_ffi.backward(tape, fr, com, gq[k], gv[k])[2]  # g_ctrl
    c = ctrl.clone().requires_grad_(True)
    q1, v1, _ = torch_ffi.step(qpos, qvel, warm, c, fr, com)
    torch.autograd.backward([q1, v1], [gq[k], gv[k]])
    ok.append(torch.allclose(g_low, c.grad, rtol=1e-5, atol=1e-6))
report['tape_replay_matches_autograd'] = ok
print('tape replayed twice, each equal to autograd:', ok)

# 4. throughput
report['throughput'] = {}
for B in (1024, 4096):
    qpos, qvel, warm, ctrl, fr, com = batch(B)
    for mode in ('forward', 'forward+backward'):
        c = ctrl.clone().requires_grad_(mode != 'forward')
        for it in range(13):
            if it == 3:
                torch.cuda.synchronize()
                t0 = time.time()
            q1, v1, _ = torch_ffi.step(qpos, qvel, warm, c, fr, com)
            if mode != 'forward':
                (q1.sum() + v1.sum()).backward()
        torch.cuda.synchronize()
        sps = 10 * B / (time.time() - t0)
        report['throughput'][f'{mode} B={B}'] = sps
        print(f'{mode:17s} B={B}: {sps:,.0f} steps/s')
report['tape_bytes_per_world_step'] = torch_ffi.tape_bytes(1)

import os  # noqa: E402

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
json.dump(report, open(args.out, 'w'), indent=1)
print('written', args.out)
