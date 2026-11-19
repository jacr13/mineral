"""Effective number of candidates of the relative best-of-K softmin, clean cost units, per tau (Phase 0 item 2)."""
import os, sys
import torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cost_scale_probe as P
from mineral.agents.diffil.best_of_k import BestOfK, OTSinkhornCriterion, SoftDTWCriterion, SequenceRegressionCriterion, SequenceCosineCriterion
dev = 'cuda:0'
objs = [('OT-cos', lambda: OTSinkhornCriterion(cost_type='cosine', step_cost='expected')),
        ('OT-L2', lambda: OTSinkhornCriterion(cost_type='l2', step_cost='expected')),
        ('DTW-cos', lambda: SoftDTWCriterion(cost_type='cosine', step_cost='expected')),
        ('index-cos', lambda: SequenceCosineCriterion()),
        ('UOTexp10-cos', lambda: OTSinkhornCriterion(cost_type='cosine', step_cost='expected', marginal_rho=10.0, marginal_side='expert'))]
for task in sys.argv[1:] or list(P.DEMOS):
    obs, full = P.load_demos(task, dev)
    flat = obs.reshape(-1, obs.shape[-1]); mean, var = flat.mean(0), flat.var(0)
    norm = lambda x: (x - mean) / torch.sqrt(var + 1e-5)
    bank = norm(obs[:8]); held = torch.where(full)[0]; held = held[held >= 8]
    g = torch.Generator().manual_seed(0)
    ids = held[torch.randint(len(held), (P.B,), generator=g)].tolist(); st = torch.randint(0, 1000 - P.T, (P.B,), generator=g).tolist()
    near = torch.stack([norm(obs[i, s:s + P.T]) for i, s in zip(ids, st)])
    far = norm(P.random_rollout(task, dev))[:, 32:64]
    for reg, aw in (('near', near), ('far', far)):
        for name, mk in objs:
            line = []
            for tau in (0.02, 0.05, 0.1, 0.2, 0.5):
                torch.manual_seed(0)
                bok = BestOfK(T=P.T, K=P.K, tau=tau, criterion=mk(), relative=True, device=dev)
                with torch.enable_grad():
                    _, info = bok(aw, bank, sim_is_window=True)
                line.append(f"{info['k_eff'].item():.2f}")
            c = info['per_step_costs']
            print(task, reg, f'{name:13s}', 'Keff@tau .02/.05/.1/.2/.5:', ' '.join(line), f'| c med {c.median().item():.3g}', flush=True)
