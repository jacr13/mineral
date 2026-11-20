"""Unit tests of the FOCUS alignment criteria: cost units, unbalanced OT, relative best-of-K, ORCA scale."""

import unittest

import torch

from mineral.agents.diffil.best_of_k import (
    BestOfK,
    ORCACriterion,
    OTSinkhornCriterion,
    SequenceRegressionCriterion,
    SoftDTWCriterion,
    log_sinkhorn,
    log_sinkhorn_unbalanced,
)

T, K, B, D = 16, 4, 6, 5


def windows(seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    sim = torch.randn(B, T, D, generator=g) * scale
    bank = torch.randn(3, 64, D, generator=g) * scale
    return sim, bank


class UnbalancedOT(unittest.TestCase):
    def test_large_rho_recovers_balanced_plan(self):
        C = torch.rand(3, T, T) * 3
        P = log_sinkhorn(C, eps=0.1, iters=200)
        for side in ('both', 'expert'):
            Pu = log_sinkhorn_unbalanced(C, eps=0.1, iters=200, rho=1e6, side=side)
            self.assertLess((P - Pu).abs().max().item(), 1e-4, side)

    def test_expert_side_keeps_every_agent_step_mass(self):
        C = torch.rand(3, T, T) * 5
        P = log_sinkhorn_unbalanced(C, eps=0.1, iters=60, rho=1.0, side='expert')
        rows = P.sum(-1) * T
        self.assertTrue(torch.allclose(rows, torch.ones_like(rows), atol=1e-5))
        self.assertAlmostEqual(P.sum(dim=(-2, -1)).mean().item(), 1.0, places=5)
        cols = P.sum(-2) * T
        self.assertGreater((cols - 1).abs().max().item(), 1e-2)  # expert side really relaxed

    def test_both_sides_total_mass_is_one(self):
        C = torch.rand(3, T, T) * 5
        P = log_sinkhorn_unbalanced(C, eps=0.1, iters=60, rho=1.0, side='both')
        self.assertTrue(torch.allclose(P.sum(dim=(-2, -1)), torch.ones(3), atol=1e-5))


class CostUnits(unittest.TestCase):
    def test_identity_plan_equals_index_wise_cost(self):
        # identical windows: OT with near-zero entropy picks the diagonal, so its expected per-step cost equals
        # the index-wise cost of a window shifted by a constant offset
        x = torch.randn(B, T, D)
        y = (x + 0.3).unsqueeze(1).expand(B, K, T, D)
        ot = OTSinkhornCriterion(eps=1e-3, iters=300, cost_type='l2', step_cost='expected')
        idx = SequenceRegressionCriterion(reduction='mean')
        Lk_ot, info_ot = ot(x, y)
        Lk_idx, info_idx = idx(x, y)
        self.assertTrue(torch.allclose(info_ot['per_step_costs'], info_idx['per_step_costs'], atol=1e-3))
        self.assertTrue(torch.allclose(Lk_ot, Lk_idx, atol=1e-3))

    def test_expected_units_do_not_depend_on_window_or_feature_size(self):
        for crit in (OTSinkhornCriterion(step_cost='expected'), SoftDTWCriterion(step_cost='expected')):
            vals = []
            for t, d in ((8, 4), (32, 4), (8, 64)):
                g = torch.Generator().manual_seed(1)
                x = torch.randn(64, t, d, generator=g)
                y = torch.randn(64, 1, t, d, generator=g)
                with torch.enable_grad():
                    _, info = crit(x, y)
                vals.append(info['per_step_costs'].mean().item())
            # random unit-variance features: mean squared difference per dimension ~ 2, minus what alignment saves
            for v in vals:
                self.assertTrue(0.5 < v < 2.5, (type(crit).__name__, vals))

    def test_legacy_units_unchanged(self):
        x = torch.randn(B, T, D)
        y = torch.randn(B, K, T, D)
        Lk, info = OTSinkhornCriterion(cost_type='l2')(x, y)
        C = torch.cdist(x.unsqueeze(1).expand(B, K, T, D).reshape(B * K, T, D), y.reshape(B * K, T, D)) ** 2
        P = log_sinkhorn(C, eps=0.1, iters=60)
        # legacy: per-step cost sum_j P_tj C_tj (summed over feature dims, rows ~1/T), window score = sum over t
        self.assertTrue(torch.allclose(info['per_step_costs'], (P * C).sum(-1).view(B, K, T), rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(Lk, info['per_step_costs'].sum(-1)))


class RelativeBestOfK(unittest.TestCase):
    def test_selection_does_not_depend_on_cost_scale(self):
        sim, bank = windows()
        out = []
        for scale in (1.0, 10.0):
            torch.manual_seed(0)
            # index-wise scores scale exactly with scale^2 (an OT plan at fixed eps would itself change)
            bok = BestOfK(T=T, K=K, tau=0.5, criterion=SequenceRegressionCriterion(reduction='mean'), relative=True)
            _, info = bok(sim * scale, bank * scale, sim_is_window=True)
            out.append(info['k_eff'].item())
        self.assertAlmostEqual(out[0], out[1], places=4)

    def test_k_eff_bounds(self):
        sim, bank = windows()
        for tau in (1e-4, 1e4):
            _, info = BestOfK(T=T, K=K, tau=tau, criterion=OTSinkhornCriterion())(sim, bank, sim_is_window=True)
            self.assertTrue(1.0 - 1e-4 <= info['k_eff'].item() <= K + 1e-4)
        self.assertAlmostEqual(info['k_eff'].item(), K, places=2)  # tau -> inf: plain average


class ORCAScale(unittest.TestCase):
    def test_normalised_reward_does_not_underflow(self):
        sim, bank = windows(scale=10.0)
        _, info = BestOfK(T=T, K=K, tau=0.5, criterion=ORCACriterion(normalize_cost=True))(sim, bank,
                                                                                           sim_is_window=True)
        self.assertGreater(torch.exp(-info['per_step_costs']).min().item(), 0.0)


if __name__ == '__main__':
    unittest.main()


class CriticTargets(unittest.TestCase):
    def test_td_lambda_matches_the_lambda_return(self):
        from types import SimpleNamespace

        from mineral.agents.diffil.focus import FOCUS

        H, B, g, lam = 32, 3, 0.99, 0.95
        torch.manual_seed(0)
        rew, nv = torch.rand(H, B), torch.rand(H, B) * 10
        done = torch.zeros(H, B)
        done[-1] = 1.0
        done[10, 1] = 1.0  # an episode end inside the window for env 1 (next value is its terminal value)
        agent = SimpleNamespace(critic_method='td-lambda', horizon_len=H, num_envs=B, gamma=g, lam=lam,
                                rew_buf=rew, next_values=nv, done_mask=done, device='cpu',
                                target_values=torch.zeros(H, B))
        FOCUS.compute_target_values(agent)

        def lam_return(t, b):
            end = next(i for i in range(t, H) if done[i, b])  # last step of this episode segment
            N = end - t + 1
            nstep = lambda n: sum(g**k * rew[t + k, b] for k in range(n)) + g**n * nv[t + n - 1, b]
            return (1 - lam) * sum(lam ** (n - 1) * nstep(n) for n in range(1, N)) + lam ** (N - 1) * nstep(N)

        for b in range(B):
            for t in (0, 5, 10, 11, 31):
                self.assertAlmostEqual(agent.target_values[t, b].item(), float(lam_return(t, b)), places=3)
