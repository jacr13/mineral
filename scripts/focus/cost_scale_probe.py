"""Phase 0 item 2: scale of every alignment objective inside the FOCUS pipeline.

For each task and objective, scores 64 agent windows (T = 32) against K = 8 candidate segments drawn from the 8
training demonstrations, exactly as BestOfK does in training, and reports:

  c      per-step cost (median, 90th percentile), the input of the reward map
  Lk     window score per candidate (median), the input of the best-of-K softmin
  dLk    spread of Lk across the K candidates (median over windows of max - min)
  Keff   effective number of candidates, 1 / sum_k w_k^2 with w = softmax(-Lk / tau), tau = 0.5
  r_exp  exp(-c) (median), and the share of steps where it is below 1e-6 (no learning signal)

Two kinds of agent windows bracket training:
  near   held-out expert windows (demonstrations 8..127 that reach full length): a converged policy
  far    rollouts of a random policy (actions tanh(N(0, 1))) from the task's stochastic initial states

Observations are normalised with the mean / variance of the 128-demonstration pool (a stand-in for the agent's
running statistics at convergence; the far windows use the same statistics, as the agent would late in training).

Run from the FOCUS checkout:  uv run --locked python <this file> [--tasks hopper ant ...] [--out file.tsv]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from mineral.agents.diffil.best_of_k import (  # noqa: E402
    BestOfK,
    ORCACriterion,
    OTSinkhornCriterion,
    SequenceCosineCriterion,
    SequenceRegressionCriterion,
    SoftDTWCriterion,
)

DEMOS = {
    'hopper': 'DFlex_hopper_demos128_return4812_len1000.pt',
    'ant': 'DFlex_ant_demos128_return9318_len1000.pt',
    'humanoid': 'DFlex_humanoid_demos128_return8116_len984.pt',
    'snu_humanoid': 'DFlex_snu_demos128_return6246_len929.pt',
}
ENVS = {'hopper': 'HopperEnv', 'ant': 'AntEnv', 'humanoid': 'HumanoidEnv', 'snu_humanoid': 'SNUHumanoidEnv'}
MM = {'hopper': 16, 'ant': 16, 'humanoid': 48, 'snu_humanoid': 8}
T, K, TAU, B = 32, 8, 0.5, 64


def objectives():
    """(name, criterion factory, feature dim or None) for every objective the pipeline offers."""
    return [
        ('OT-L2 feat64 (paper OT-L2)', lambda: OTSinkhornCriterion(cost_type='l2'), 64),
        ('OT-L2 identity', lambda: OTSinkhornCriterion(cost_type='l2'), None),
        ('OT-cos identity (paper OT-Cos)', lambda: OTSinkhornCriterion(cost_type='cosine'), None),
        ('OT-cos feat64', lambda: OTSinkhornCriterion(cost_type='cosine'), 64),
        ('L2 index feat64 (paper FOCUS-L2)', lambda: SequenceRegressionCriterion(reduction='mean'), 64),
        ('L2 index identity', lambda: SequenceRegressionCriterion(reduction='mean'), None),
        ('cos index identity', lambda: SequenceCosineCriterion(), None),
        ('softDTW-L2 feat64', lambda: SoftDTWCriterion(cost_type='l2'), 64),
        ('softDTW-cos identity', lambda: SoftDTWCriterion(cost_type='cosine'), None),
        ('subseq softDTW-L2 feat64', lambda: SoftDTWCriterion(cost_type='l2', subsequence=True, open_end=True), 64),
        ('ORCA-L2 feat64 (port)', lambda: ORCACriterion(cost_type='l2'), 64),
    ]


def load_demos(task, device):
    d = torch.load(os.path.join('experts/demos', DEMOS[task]), map_location=device)
    obs = d['obs']['obs'] if isinstance(d['obs'], dict) else d['obs']
    done = d['done']
    full = torch.tensor([not bool(done[i, :-1].any()) for i in range(done.shape[0])], device=device)
    return obs.float(), full


def random_rollout(task, device, steps=64, seed=0):
    import dflex.envs as E

    env = getattr(E, ENVS[task])(
        num_envs=B, device=device, seed=seed, episode_length=1000, MM_caching_frequency=MM[task],
        stochastic_init=True, no_grad=True, render=False,
    )
    obs = env.reset()
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            a = torch.tanh(torch.randn(B, env.num_actions, device=device, generator=g))
            obs, _, _, info = env.step(a)
            o = info.get('obs_before_reset', obs)
            out.append((obs if o is None else o).clone())
    return torch.stack(out, 1)  # [B, steps, d]


def probe(name, make_crit, feat, agent_w, bank, seed):
    torch.manual_seed(seed)
    d = bank.shape[-1]
    bok = BestOfK(T=T, K=K, tau=TAU, mlp_features_dim=feat, feature_dim=d, criterion=make_crit(),
                  device=bank.device)
    with torch.no_grad() if 'softDTW' not in name else torch.enable_grad():
        _, info = bok(agent_w, bank, sim_is_window=True)
    c = info['per_step_costs'].detach()
    Lk = info['Lk'].detach()
    w = torch.softmax(-Lk / TAU, dim=1)
    keff = 1.0 / (w ** 2).sum(1)
    dlk = Lk.max(1).values - Lk.min(1).values
    r = torch.exp(-c)
    q = lambda x, p: float(torch.quantile(x.flatten().float(), p))
    return dict(c_med=q(c, .5), c_p90=q(c, .9), Lk_med=q(Lk, .5), dLk_med=q(dlk, .5), Keff_med=q(keff, .5),
                rexp_med=q(r, .5), rexp_dead=float((r < 1e-6).float().mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=list(DEMOS))
    ap.add_argument('--seeds', nargs='+', type=int, default=[10, 20, 30])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rows = []
    for task in args.tasks:
        obs, full = load_demos(task, device)
        mean, var = obs.reshape(-1, obs.shape[-1]).mean(0), obs.reshape(-1, obs.shape[-1]).var(0)
        norm = lambda x: (x - mean) / torch.sqrt(var + 1e-5)
        bank = norm(obs[:8])
        held = torch.where(full)[0]
        held = held[held >= 8]
        g = torch.Generator(device='cpu').manual_seed(0)
        ids = held[torch.randint(len(held), (B,), generator=g)].to(device)
        st = torch.randint(0, 1000 - T, (B,), generator=g).to(device)
        near = torch.stack([norm(obs[i, s:s + T]) for i, s in zip(ids.tolist(), st.tolist())])
        far_full = norm(random_rollout(task, device))
        far = far_full[:, 32:64]  # past the first half-window, as the agent's windows start anywhere
        for regime, aw in (('near', near), ('far', far)):
            for name, mk, feat in objectives():
                for seed in (args.seeds if feat else args.seeds[:1]):
                    r = probe(name, mk, feat, aw, bank, seed)
                    rows.append(dict(task=task, regime=regime, objective=name, seed=seed, **r))
                    print(task, regime, f'{name:34s}', seed, ' '.join(f'{k}={v:.3g}' for k, v in r.items()), flush=True)
    if args.out:
        keys = list(rows[0])
        with open(args.out, 'w') as f:
            f.write('\t'.join(keys) + '\n')
            for r in rows:
                f.write('\t'.join(str(r[k]) for k in keys) + '\n')


if __name__ == '__main__':
    main()
