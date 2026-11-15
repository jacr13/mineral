"""Reset DroneEnv to the demos' first states and replay the stored actions open loop (env resettable to given states)."""

import torch

from mineral.envs.drone import DroneEnv, state_from_obs

d = torch.load('experts/demos/Warp_Drone_demos128_return251_len300.pt')
n = 16
env = DroneEnv(n, 'cuda:0', no_grad=True)
q, qd = state_from_obs(d['obs']['obs'][:n, 0].cuda())
env.set_state(torch.arange(n, device='cuda:0'), q, qd, torch.zeros(n, 3, device='cuda:0'))
o = env.obs()
err, rew = [], []
for t in range(300):
    err.append((o - d['obs']['obs'][:n, t].cuda()).abs().max().item())
    o2, r, done, info = env.step(d['act'][:n, t].cuda())
    rew.append((r.cpu() - d['rew'][:n, t]).abs().max().item())
    o = o2['obs']
print(
    'open-loop replay max |obs - demo obs| at t=0,10,50,100,299:',
    [f'{err[t]:.1e}' for t in (0, 10, 50, 100, 299)],
    f'max |rew diff| {max(rew):.1e}',
)
