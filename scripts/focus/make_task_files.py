"""Generate FOCUS task files from one design (no task file is edited by hand).

Every arm is the clean default plus at most one changed setting, except the reference arm, which is the paper's
FOCUS-OT-Cos configuration as it ran. Writes tasks/focus/<batch>/<task>_<arm>.yaml in the FOCUS checkout.

usage: python make_task_files.py <batch> [arm ...]   (run from the FOCUS checkout)
"""

import copy
import os
import sys

import yaml

TASKS = {
    'hopper': ('DFlex_hopper_demos128_return4812_len1000.pt', 10_000_000),
    'ant': ('DFlex_ant_demos128_return9318_len1000.pt', 10_000_000),
    'humanoid': ('DFlex_humanoid_demos128_return8116_len984.pt', 15_000_000),
    'snu_humanoid': ('DFlex_snu_demos128_return6246_len929.pt', 15_000_000),
}

# The clean default: one setting per line, the same for every task and every variant.
DEFAULT = dict(
    imitation_loss_type='ot',  # alignment: ot | index (no alignment) | softdtw | softdtw_subseq
    input_type='state',  # matched item: the state (DFlex states hold velocities)
    loss_mlp_features_dim=None,  # no feature map: normalised states
    loss_ot_cost_type='cosine',  # ground cost
    loss_cost_units='expected',  # per-step cost = expected ground cost of the step's match; window score = mean
    loss_best_of_k_k=8,
    loss_best_of_k_relative=True,  # softmin on scores relative to the K candidates
    loss_best_of_k_tau=0.1,
    loss_ot_eps=0.1,
    loss_ot_iters=60,
    loss_ot_detach_plan=True,  # fixed coupling
    loss_ot_marginal_rho=None,
    loss_ot_marginal_side='both',
    critic_reward_mapping='log_exp',
    critic_reward_minus_cost=True,  # reward = phi(c) - c, for every mapping
    critic_reward_shapping=True,
    critic_disabled=False,
    critic_reward_normalize=False,
    loss_use_huber_speedup=False,
    loss_use_detached_prev_obs=False,
    actor_grad_estimator='pathwise',
)

# The paper's FOCUS-OT-Cos as it ran (audit/v2/runs.jsonl): legacy units, absolute softmin tau 0.5, state pairs on
# Ant. critic_reward_minus_cost null applies -c to log_exp, as in those runs.
REFERENCE = dict(DEFAULT, loss_cost_units='legacy', loss_best_of_k_relative=False, loss_best_of_k_tau=0.5,
                 critic_reward_minus_cost=None)

ARMS = {
    'ref_otcos': {},  # REFERENCE, see below
    'default': {},
    'cost_l2': dict(loss_ot_cost_type='l2'),
    'noalign': dict(imitation_loss_type='index'),
    'dtw': dict(imitation_loss_type='softdtw'),
    'dtw_subseq': dict(imitation_loss_type='softdtw_subseq'),
    'k1': dict(loss_best_of_k_k=1),
    'noalign_k1': dict(imitation_loss_type='index', loss_best_of_k_k=1),
    'feat64': dict(loss_mlp_features_dim=64),
    'statepairs': dict(input_type='state_state'),
    'no_minus_c': dict(critic_reward_minus_cost=False),
    'map_exp': dict(critic_reward_mapping='exp'),
    'legacy_units': dict(loss_cost_units='legacy'),
    'abs_softmin': dict(loss_best_of_k_relative=False, loss_best_of_k_tau=0.5),
    'through_sinkhorn': dict(loss_ot_detach_plan=False),
    'score_function': dict(actor_grad_estimator='score_function'),
}
# second screening round: the default without the -c term (reward never negative), and the L2 ablation of it
ARMS['no_minus_c_cost_l2'] = dict(critic_reward_minus_cost=False, loss_ot_cost_type='l2')
# third round: the two other changes of the clean default, each against the default without -c
ARMS['no_minus_c_legacy_units'] = dict(critic_reward_minus_cost=False, loss_cost_units='legacy')
ARMS['no_minus_c_abs_softmin'] = dict(critic_reward_minus_cost=False, loss_best_of_k_relative=False,
                                      loss_best_of_k_tau=0.5)
for rho in (0.03, 0.1, 0.3, 1.0, 3.0):
    ARMS[f'uot_both_rho{rho:g}'] = dict(loss_ot_marginal_rho=rho, loss_ot_marginal_side='both')
    ARMS[f'uot_expert_rho{rho:g}'] = dict(loss_ot_marginal_rho=rho, loss_ot_marginal_side='expert')

# The method fixed after Phase 0 (Joao, 2026-11-20): the clean default without the -c term. Arms 'f_*' are this
# default plus at most one changed setting; '_n_trajs' sets the number of demonstrations.
FINAL = dict(DEFAULT, critic_reward_minus_cost=False)
F_ARMS = {
    'f_default': {},
    'f_statepairs': dict(input_type='state_state'),
    'f_demos1': dict(_n_trajs=1),
    'f_demos2': dict(_n_trajs=2),
    'f_demos4': dict(_n_trajs=4),
    'f_noalign': dict(imitation_loss_type='index'),
    'f_dtw': dict(imitation_loss_type='softdtw'),
    'f_dtw_subseq': dict(imitation_loss_type='softdtw_subseq'),
    'f_k1': dict(loss_best_of_k_k=1),
    'f_noalign_k1': dict(imitation_loss_type='index', loss_best_of_k_k=1),
    'f_cost_l2': dict(loss_ot_cost_type='l2'),
    'f_feat64': dict(loss_mlp_features_dim=64),
    'f_minus_c': dict(critic_reward_minus_cost=True),
    'f_through_sinkhorn': dict(loss_ot_detach_plan=False),
    'f_score_function': dict(actor_grad_estimator='score_function'),
}
for rho in (0.03, 0.1, 0.3, 1.0, 3.0):
    F_ARMS[f'f_uot_expert_rho{rho:g}'] = dict(loss_ot_marginal_rho=rho, loss_ot_marginal_side='expert')
    F_ARMS[f'f_uot_both_rho{rho:g}'] = dict(loss_ot_marginal_rho=rho, loss_ot_marginal_side='both')
ARMS.update({k: v for k, v in F_ARMS.items()})

# Run-level settings, the same for every arm
EVAL_EVERY = 500_000


def task_file(batch, task, arm):
    demo, budget = TASKS[task]
    base = REFERENCE if arm == 'ref_otcos' else FINAL if arm.startswith('f_') else DEFAULT
    focus = copy.deepcopy(base)
    if arm == 'ref_otcos' and task == 'ant':
        focus['input_type'] = 'state_state'
    change = dict(ARMS[arm])
    n_trajs = change.pop('_n_trajs', 8)
    focus.update(change)
    focus = dict(demos=dict(path='experts/demos/' + demo, n_trajs=n_trajs), **focus)
    return dict(
        task=dict(name='DFlex', env=dict(env_name=task)),
        agent=dict(
            name='FOCUS/DFlexAntSAPO',
            ckpt_every=-1,  # no periodic checkpoints: final.pth (= at budget) only
            protocol_eval=True,  # shared evaluation of every method (Agent.protocol_*), on a dedicated env
            eval_every_steps=EVAL_EVERY,
            save_eval_rollouts=False,
            shac=dict(max_epochs=50000, max_agent_steps=budget),
            focus=focus,
            network=dict(actor_kwargs=dict(mlp_kwargs=dict(units=[128, 64, 32])),
                         critic_kwargs=dict(mlp_kwargs=dict(units=[64, 64]))),
        ),
        logdir=f'workdir/focus-{batch}-{task}-{arm}/$(date +%Y%m%d-%H%M%S.%2N)',
        wandb=dict(mode='offline'),
        run='train_eval',
        seed=10,
        sweep=dict(seed=[10, 20, 30]),
        spawner=dict(runtime='12h'),
    )


# ---------------------------------------------------------------------------------------------------------------
# Baselines: the method's own task file (tasks/<method>/dflex_<task>.yaml) plus the rerun protocol. Arms are named
# bl_<method>[_<variant>]; a variant sets the one tuning knob of that method (the tuning rule of the shared protocol).
# ---------------------------------------------------------------------------------------------------------------
BASELINES = {
    # method: (source task folder, budget key under agent, {variant: {dotted key: value}}, reference row?)
    'gaifo': ('gaifo', 'ppo', {'gp1': {'agent.il.discriminator.grad_penalty': 1.0},
                               'gp10': {'agent.il.discriminator.grad_penalty': 10.0}}, False),
    'maad': ('maad', 'ppo', {'gp1': {'agent.il.discriminator.grad_penalty': 1.0},
                             'gp10': {'agent.il.discriminator.grad_penalty': 10.0}}, False),
    'gail': ('gail', 'ppo', {'gp1': {'agent.il.discriminator.grad_penalty': 1.0},
                             'gp10': {'agent.il.discriminator.grad_penalty': 10.0}}, True),
    'dac': ('dac', 'sac', {}, False),
    'opolo': ('opolo', 'sac', {}, False),
    'pwil': ('pwil', 'sac', {'b2': {'agent.pwil.beta': 2.0}, 'b5': {'agent.pwil.beta': 5.0}}, False),
    'lwail': ('lwail', 'ddpg', {}, False),
    'oops': ('oops', 'ddpg', {'it1000': {'agent.oops.sinkhorn_iters': 1000}}, False),
    # ILD as in its paper (whole episodes, truncated gradients); both rollouts of an iteration are counted
    'ild': ('ild/paper', 'shac', {
        # as in its paper: whole episodes, gradient truncated every 10 (or 32) steps; both rollouts counted
        'paper': {'agent.ild.mode': 'paper', 'agent.ild.count_all_sim_steps': True},
        'paper_t32': {'agent.ild.mode': 'paper', 'agent.ild.count_all_sim_steps': True, 'agent.ild.trunc_len': 32},
        # the earlier window variant: SHAC-style 32-step windows (FOCUS's horizon), SAPO or SHAC base
        'win_sapo': {'_src': 'ild', 'agent.name': 'ILD/DFlexAntSAPO', 'agent.ild.mode': 'window'},
        'win_shac': {'_src': 'ild', 'agent.name': 'ILD/DFlexAntSHAC', 'agent.ild.mode': 'window'},
    }, False),
    'diffmimic': ('diffmimic', 'diffmimic', {}, False),
    'bco': ('bco', None, {}, False),
    'bc': ('bc', None, {}, True),
}


def _set_dotted(cfg, key, value):
    parts = key.split('.')
    d = cfg
    for k in parts[:-1]:
        d = d.setdefault(k, {})
    d[parts[-1]] = value


def baseline_task_file(batch, task, arm):
    _, rest = arm.split('_', 1)
    method, _, variant = rest.partition('_')
    src, fam, variants, _ = BASELINES[method]
    vset = dict(variants.get(variant, {})) if variant else {}
    src = vset.pop('_src', src)
    cfg = yaml.safe_load(open(os.path.join('tasks', src, f'dflex_{task}.yaml')))
    sweep = cfg.pop('sweep', {}) or {}
    for k, v in sweep.items():  # scalar sweep settings belong to the method: fold them into the body
        if k in ('seed', 'logdir'):
            continue
        if isinstance(v, list):
            if not variant or not any(k in vv for vv in [variants.get(variant, {})]):
                raise ValueError(f'{method}: sweep list {k}={v} needs an explicit variant')
            continue
        _set_dotted(cfg, k, v)
    if variant:
        for k, v in vset.items():
            _set_dotted(cfg, k, v)
    elif variants and method not in ('oops', 'ild'):
        raise ValueError(f'{method}: choose a variant from {list(variants)}')
    budget = TASKS[task][1]
    ag = cfg['agent']
    if fam is not None:
        ag.setdefault(fam, {})['max_agent_steps'] = budget
    ag['ckpt_every'] = -1
    ag['protocol_eval'] = True
    ag['eval_every_steps'] = EVAL_EVERY
    if method == 'ild':
        ag['save_eval_rollouts'] = False
    cfg['task']['env'].pop('numEnvs', None)  # 64, the task default, for every method
    cfg['logdir'] = f'workdir/focus-{batch}-{task}-{arm}/$(date +%Y%m%d-%H%M%S.%2N)'
    cfg['wandb'] = dict(mode='offline')
    cfg['run'] = 'train_eval'
    cfg['seed'] = 10
    cfg['sweep'] = dict(seed=[10, 20, 30])
    cfg['spawner'] = dict(runtime='12h')
    return cfg


def main():
    argv = sys.argv[1:]
    tasks = list(TASKS)
    if '--tasks' in argv:
        i = argv.index('--tasks')
        argv, tasks = argv[:i], argv[i + 1:]
    batch, arms = argv[0], argv[1:] or list(ARMS)
    out = os.path.join('tasks/focus', batch)
    os.makedirs(out, exist_ok=True)
    for task in tasks:
        for arm in arms:
            path = os.path.join(out, f'{task}_{arm}.yaml')
            body = baseline_task_file(batch, task, arm) if arm.startswith('bl_') else task_file(batch, task, arm)
            hdr = (f'# Generated by scripts/focus/make_task_files.py ({batch}, arm {arm}); do not edit.\n'
                   f'# Settings are in the body, not in the sweep block (--set seed replaces the sweep block).\n')
            with open(path, 'w') as f:
                f.write(hdr + yaml.safe_dump(body, sort_keys=False, default_flow_style=None))
    print(f'{len(tasks) * len(arms)} task files in {out}')


if __name__ == '__main__':
    main()
