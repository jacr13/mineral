"""Phase 0 diagnosis: label the same learner states with several teachers and compare them with the gradient teacher.

Runs the normal entry point (mineral.scripts.run) with MENTOR.train replaced: Stage I, cloning and `--warm_rounds`
ordinary MENTOR rounds (gradient teacher) give a learner; then `--n` visited states x and neighbours
x' = x + eps * (demo std) * N(0, I) are labelled by each teacher variant. Per variant:
  cost_q50       median rejoin cost of the label's plan (the teacher's own objective)
  d_u0_ref       median ||u0 - u0_ref|| / sqrt(A), distance of the first action to the gradient teacher's
  corr_ref       median cosine between the corrections (u0 - u0_warm) and (u0_ref - u0_warm)
  spread         median ||u0(x) - u0(x')|| / sqrt(A) across neighbouring states (the diagnosis)
  steps_per_lab  simulator steps per label (forward; forward+backward for the gradient teacher, listed apart)
  sec            wall-clock of labelling x and x'

usage (from the checkout):
  python scripts/tutor/teacher_probe.py --out probe.json --env ant --warm_rounds 2 -- <hydra overrides>
Variants are given as name:key=val,key=val (teacher keys); the first variant must be the reference.
"""

import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '../..'))
sys.path.insert(0, CODE)

DEFAULT_VARIANTS = [
    'adam:kind=rejoin',
    'warm:kind=rejoin,iters=0',
    'fdadam:kind=fdadam,fd_mode=full',
    'spsa16:kind=fdadam,fd_mode=rand,fd_dirs=16',
    'lmfd5:kind=lmplan,lm_jacobian=fd,iters=5',
    'lmbr10:kind=lmplan,lm_jacobian=broyden,iters=10',
    'mppi:kind=mppi,sample_n=32,sample_noise=0.05,iters=10',
    'mppi_pull:kind=mppi,sample_n=32,sample_noise=0.05,iters=10,mppi_pull=1.0',
    'cem:kind=sample,sample_mode=cem,sample_n=64,sample_noise=0.05,sample_iters=5',
]


def parse_val(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {'true': True, 'false': False}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--warm_rounds', type=int, default=2)
    ap.add_argument('--n', type=int, default=340, help='states labelled (one teacher batch)')
    ap.add_argument('--eps', type=float, default=0.05, help='neighbour distance, in demo standard deviations')
    ap.add_argument('--variants', nargs='*', default=DEFAULT_VARIANTS)
    ap.add_argument('hydra', nargs='*')
    args = ap.parse_args()

    import torch
    from omegaconf import OmegaConf

    from mineral.agents.diffil import mentor as M
    from mineral.agents.diffil.mentor_gf import GF_TEACHERS
    from mineral.agents.diffil.mentor_sim import SimCounter

    KINDS = {'rejoin': M.RejoinTeacher, 'sample': M.SampleTeacher, 'ilqr': M.ILQRTeacher, **GF_TEACHERS}

    def probe(self):
        cfg = self.mentor_config
        OmegaConf.set_struct(cfg, False)
        cfg.rounds = args.warm_rounds
        orig_train(self)  # Stage I + cloning + warm rounds with the configured (gradient) teacher
        torch.manual_seed(1234)
        visited = self.collect()
        idx = torch.randperm(visited.shape[0], generator=torch.Generator(device='cpu').manual_seed(0))[: args.n]
        x = visited[idx.to(visited.device)]
        std = self.S[self.valid].std(0)
        g = torch.Generator(device=x.device).manual_seed(1)
        xn = x + args.eps * std * torch.randn(x.shape, device=x.device, generator=g)
        base = OmegaConf.to_container(cfg.teacher, resolve=True)
        base['batch'] = args.n
        A = self.spec.act_dim
        rows, ref = [], {}
        for spec_ in args.variants:
            name, _, kv = spec_.partition(':')
            over = {k: parse_val(v) for k, v in (p.split('=', 1) for p in kv.split(',') if p)}
            tc = copy.deepcopy(base)
            tc.update(over)
            kind = tc.get('kind', 'rejoin')
            counter = SimCounter()
            T = KINDS[kind](self.spec, self.S, self.S1, self.A_hat, self.lengths, args.n, self.device, counter,
                            OmegaConf.create(tc))
            T.anchor_to(self.policy)
            f0, fb0 = counter.fwd, counter.fwdbwd  # building the teacher steps nothing; kept for safety
            torch.cuda.synchronize()
            t0 = time.time()
            u, c = T.label(x)
            un, _ = T.label(xn)
            torch.cuda.synchronize()
            sec = time.time() - t0
            fwd, fb = (counter.fwd - f0) / (2 * args.n), (counter.fwdbwd - fb0) / (2 * args.n)
            u, un = u.clamp(-1, 1), un.clamp(-1, 1)
            if name == 'adam':
                ref['u'] = u
            if name == 'warm':
                ref['w'] = u
            row = dict(name=name, kind=kind, over=over, cost_q50=c.median().item(), cost_mean=c.mean().item(),
                       spread=((u - un).norm(dim=-1) / A**0.5).median().item(),
                       fwd_per_label=fwd, fwdbwd_per_label=fb, sec=sec)
            if 'u' in ref:
                row['d_u0_ref'] = ((u - ref['u']).norm(dim=-1) / A**0.5).median().item()
            if 'u' in ref and 'w' in ref:
                d1, d2 = u - ref['w'], ref['u'] - ref['w']
                row['corr_ref'] = torch.nn.functional.cosine_similarity(d1, d2, dim=-1).median().item()
                row['corr_size'] = (d1.norm(dim=-1) / d2.norm(dim=-1).clamp_min(1e-9)).median().item()
            rows.append(row)
            print('[probe]', json.dumps(row), flush=True)
            del T
            torch.cuda.empty_cache()
        meta = dict(env=self.env_name, warm_rounds=args.warm_rounds, n=args.n, eps=args.eps, act_dim=A,
                    gpu=torch.cuda.get_device_name(0), stage1_steps=self.counters['stage1'].total)
        json.dump(dict(meta=meta, rows=rows), open(args.out, 'w'), indent=1)
        print(f'[probe] wrote {args.out}', flush=True)
        os._exit(0)  # skip the final evaluation of the entry point

    orig_train = M.MENTOR.train
    M.MENTOR.train = probe
    sys.argv = ['run'] + args.hydra
    import mineral.agents as _agents
    from mineral.agents.diffil.tutor import TUTOR as _TUTOR

    _agents.TUTOR = _TUTOR  # TUTOR agent (scripts/tutor/run.py does the same)
    import runpy

    runpy.run_module('mineral.scripts.run', run_name='__main__')


if __name__ == '__main__':
    main()
