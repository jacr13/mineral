"""MENTOR simulator maps, Stage I inversion and rejoin teacher on real DFlex envs (GPU only).

The stored demo actions are read here only to check the simulator maps and the inversion;
the MENTOR agent itself never reads them.
"""

import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEMOS = {
    'hopper': ROOT / 'experts/demos/DFlex_hopper_demos128_return4812_len1000.pt',
    'ant': ROOT / 'experts/demos/DFlex_ant_demos128_return9318_len1000.pt',
    'humanoid': ROOT / 'experts/demos/DFlex_humanoid_demos128_return8116_len984.pt',
}
DEV = 'cuda:0'


def _load(env, n_traj=1):
    d = torch.load(DEMOS[env], map_location=DEV)
    return d['obs']['obs'][:n_traj], d['next_obs']['obs'][:n_traj], d['act'][:n_traj]


@unittest.skipUnless(torch.cuda.is_available(), 'needs a CUDA device for DFlex')
class MentorSimTests(unittest.TestCase):
    def _step(self, env, n):
        from mineral.agents.diffil.mentor_sim import OneStep, SimCounter, get_spec

        spec = get_spec(env)
        return spec, OneStep(spec, n, DEV, SimCounter())

    def test_maps_reproduce_demo_transitions(self):
        for env, path in DEMOS.items():
            if not path.exists():
                continue
            with self.subTest(env=env):
                n = 256
                spec, step = self._step(env, n)
                obs, nxt, act = _load(env)
                D = spec.obs_dim
                s, s1, a = obs[0, 100 : 100 + n, :D], nxt[0, 100 : 100 + n, :D], act[0, 100 : 100 + n]
                pred = step(s, a.clamp(-1, 1), grad=False)
                rel = (pred - s1).norm(dim=-1) / (s1 - s).norm(dim=-1).clamp_min(1e-9)
                self.assertLess(rel.median().item(), 1e-3, f'{env}: rel residual median {rel.median().item():.2e}')
                self.assertEqual(step.counter.fwd, n)

    def test_step_is_differentiable(self):
        spec, step = self._step('hopper', 64)
        obs, nxt, _ = _load('hopper')
        a = torch.zeros(64, spec.act_dim, device=DEV, requires_grad=True)
        loss = ((step(obs[0, :64, :11], a) - nxt[0, :64, :11]) ** 2).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(a.grad.abs().sum().item(), 0)
        self.assertEqual(step.counter.fwdbwd, 64)

    def test_lm_inversion_recovers_actions(self):
        from mineral.agents.diffil.mentor import lm_invert

        n = 512
        spec, step = self._step('hopper', n)
        obs, nxt, act = _load('hopper')
        s, s1 = obs[0, :n, :11], nxt[0, :n, :11]
        w = 1.0 / (s1 - s).std(0)
        a_hat, cost = lm_invert(step, s, s1, w, iters=5, starts=1)
        err = (a_hat - act[0, :n].clamp(-1, 1)).abs()
        self.assertLess(err.median().item(), 1e-3)
        self.assertLess(cost.median().item(), 1e-6)

    def test_rejoin_teacher_on_demo_states(self):
        from omegaconf import OmegaConf

        from mineral.agents.diffil.mentor import RejoinTeacher
        from mineral.agents.diffil.mentor_sim import SimCounter, get_spec

        spec = get_spec('hopper')
        obs, nxt, act = _load('hopper', n_traj=3)
        S, S1 = obs[..., :11].contiguous(), nxt[..., :11].contiguous()
        A = act.clamp(-1, 1)  # stand-in for Stage I output
        lengths = torch.full((3,), S.shape[1], device=DEV, dtype=torch.long)
        cfg = OmegaConf.create(
            {
                'horizon': 8,
                'num_windows': 3,
                'screen': True,
                'iters': 5,
                'lr': 0.2,
                'action_reg': 0.05,
                'phase_offsets': [-2, 0, 2],
                'batch': 32,
            }
        )
        counter = SimCounter()
        teacher = RejoinTeacher(spec, S, S1, A, lengths, 32, DEV, counter, cfg)
        x = S[0, 200:232] + 0.01 * torch.randn(32, 11, device=DEV)
        labels, cost = teacher.label(x)
        self.assertEqual(labels.shape, (32, 3))
        self.assertTrue(((labels >= -1) & (labels <= 1)).all())
        self.assertLess(cost.median().item(), 1.0)
        # screening (3 windows x 8 steps forward) + 5 Adam iterations and a final forward on the best window
        self.assertEqual(counter.fwd, 32 * 3 * 8 + 32 * 8)
        self.assertEqual(counter.fwdbwd, 32 * 8 * 5)


if __name__ == '__main__':
    unittest.main()
