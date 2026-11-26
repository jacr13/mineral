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
    # added 2026-11-22 (Joao: the other SHAC-paper tasks); expert: SAPO seed 5020 (tasks/focus/experts), not on the
    # cluster's input folder: launch its cells with --attach_input experts/demos=<dir holding this file>
    'cheetah': ('DFlex_cheetah_demos128_return8078_len1000.pt', 10_000_000),
    # added 2026-11-23 (Joao): 4 M steps as the SAPO config; 240-step episodes; scored random-referenced (table.py)
    'cartpoleswingup': ('DFlex_cartpoleswingup_demos128_return-286_len240.pt', 4_000_000),
}
# Phase 5 ordering task (Rewarped soft Jumper, gradsim): kept apart from TASKS so DFlex batches never include it.
# Network, encoder keys, env count and critic settings as in the earlier FOCUS Jumper runs
# (tasks/focus/rewarped_soft_jumper.yaml); the objective is the clean FOCUS config like every DFlex task.
REWARPED_TASKS = {
    'jumper': ('Rewarped_Jumper_demos64_return2046_len300.pt', 6_200_000),
}
JUMPER_AGENT = dict(
    network=dict(encoder_kwargs=dict(mlp_keys='com_q|com_qd|actions'),
                 actor_kwargs=dict(mlp_kwargs=dict(units=[512, 256])),
                 critic_kwargs=dict(mlp_kwargs=dict(units=[256, 256]))),
    shac=dict(critic_optim_kwargs=dict(lr=5e-4), target_critic_alpha=0.995),
)

# Ordering test (Phase 5, option b, Joao 2026-11-22): the imitation cost on positions only (dflex/envs/<task>.py
# observation layouts). Without velocities, a gait and the same gait played backwards visit the same postures, so the
# order of the states is the only cue for the direction of motion. Actor and critic still see the full observation.
POSITION_DIMS = {
    'hopper': [0, 1, 2, 3, 4],  # joint_q[1:]: height, pitch, 3 joints (5:11 = joint_qd)
    'cheetah': [0, 1, 2, 3, 4, 5, 6, 7],  # joint_q[1:]: height, pitch, 6 joints (8:17 = joint_qd)
    'ant': [0, 1, 2, 3, 4] + list(range(11, 19)) + [27, 28],  # height, rotation, 8 joints, up, heading
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
    # SAPO entropy terms: legacy (default, every earlier run) = no actor entropy, alpha fixed, constant +0.5 in the
    # critic targets; none = no entropy and no bonus; sapo = SAPO's maximum-entropy objective
    'f_ent_none': dict(backbone_entropy='none'),
    'f_ent_sapo': dict(backbone_entropy='sapo'),
    # SHAC-style backbone (Gaussian policy, ELU, single critic + target, Adam, no entropy), critic targets fixed (91a1a9d)
    'f_shac': dict(_agent='FOCUS/DFlexAntSHAC'),
}
# p0screen8 as it ran: these 'fn_' names do not start with 'f_', so task_file() gave them the DEFAULT base, i.e. WITH
# the -c reward term, while their control f_ent_none has none (found 2026-11-21: round 8 confounded). Kept unchanged
# so the files match the runs; the corrected screen is p0screen8b (f_n_uot_* below).
FN_ON_DEFAULT_AS_RAN = set()
for rho in (0.1, 0.3, 1.0):
    FN_ON_DEFAULT_AS_RAN.add(f'fn_uot_expert_rho{rho:g}')
    F_ARMS[f'fn_uot_expert_rho{rho:g}'] = dict(backbone_entropy='none', loss_ot_marginal_rho=rho,
                                               loss_ot_marginal_side='expert')
# corrected round 8 (p0screen8b): expert-side UOT on the clean FINAL config with the no-entropy backbone
for rho in (0.1, 0.3, 1.0):
    F_ARMS[f'f_n_uot_rho{rho:g}'] = dict(backbone_entropy='none', loss_ot_marginal_rho=rho,
                                         loss_ot_marginal_side='expert')
# Phase 3 FOCUS variants (p3main): clean FINAL config, no-entropy backbone (round 7); UOT added once p0screen8b fixes rho
F_ARMS['f_main_ot'] = dict(backbone_entropy='none')
F_ARMS['f_main_dtw'] = dict(backbone_entropy='none', imitation_loss_type='softdtw')
F_ARMS['f_main_noalign'] = dict(backbone_entropy='none', imitation_loss_type='index')
# Phase 4 ablations (in p3main, control f_main_ot): one setting changed against f_main_ot each
_N = dict(backbone_entropy='none')
F_ARMS.update({
    # 1. gradient path
    'f_ab_through_sinkhorn': dict(_N, loss_ot_detach_plan=False),
    'f_ab_score_function': dict(_N, actor_grad_estimator='score_function'),
    # 2. alignment objective at the default cost (soft-DTW and no alignment are f_main_dtw / f_main_noalign)
    'f_ab_dtw_subseq': dict(_N, imitation_loss_type='softdtw_subseq'),
    # 3. best-of-K (K = 8 is the control)
    'f_ab_k1': dict(_N, loss_best_of_k_k=1),
    'f_ab_k2': dict(_N, loss_best_of_k_k=2),
    'f_ab_k4': dict(_N, loss_best_of_k_k=4),
    'f_ab_k16': dict(_N, loss_best_of_k_k=16),
    'f_ab_noalign_k1': dict(_N, imitation_loss_type='index', loss_best_of_k_k=1),
    # 5. reward transform and critic; backbone
    'f_ab_minus_c': dict(_N, critic_reward_minus_cost=True),
    'f_ab_rew_neg': dict(_N, critic_reward_mapping='neg'),
    'f_ab_rew_exp': dict(_N, critic_reward_mapping='exp'),
    'f_ab_critic_off': dict(_N, critic_disabled=True),
    'f_ab_shac': dict(_N, _agent='FOCUS/DFlexAntSHAC'),
    'f_ab_ent_sapo': dict(backbone_entropy='sapo'),
    # demonstrations, matched item, cost, feature map (phase-0 defaults kept, the alternatives as one-setting arms)
    'f_ab_demos1': dict(_N, _n_trajs=1),
    'f_ab_demos2': dict(_N, _n_trajs=2),
    'f_ab_demos4': dict(_N, _n_trajs=4),
    'f_ab_demos16': dict(_N, _n_trajs=16),  # few-demonstration study (R16, Joao 2026-11-25)
    'f_ab_statepairs': dict(_N, input_type='state_state'),
    'f_ab_cost_l2': dict(_N, loss_ot_cost_type='l2'),
    'f_ab_feat64': dict(_N, loss_mlp_features_dim=64),
})
for rho in (0.3, 1.0, 3.0, 10.0, 100.0):
    F_ARMS[f'f_ab_uot_rho{rho:g}'] = dict(_N, loss_ot_marginal_rho=rho, loss_ot_marginal_side='expert')
# 1b. score-function arm with its tuned actor learning rate (p4sftune: 2e-4 best of 2e-3 ... 2e-5, 2026-11-22)
F_ARMS['f_ab_score_function_lr2e-4'] = dict(_N, actor_grad_estimator='score_function',
                                            _shac=dict(actor_optim_kwargs=dict(lr=2e-4, betas=[0.7, 0.95])))
# best-of-K selection (Joao, 2026-11-23): hard min (tau -> 0, k_eff ~ 1) and plain average (tau large, k_eff ~ K)
# against the relative softmin tau = 0.1 of f_main_ot
# actor bootstrap fixed (writer request R19, 2026-11-25): FOCUS-OT with the value after the last action, zero after a
# fall; every other FOCUS arm ran with actor_bootstrap 'as_run'. Paired check against f_main_ot on the six tasks.
F_ARMS['f_fix_ot'] = dict(_N, actor_bootstrap='next_state')
F_ARMS['f_ab_kmin'] = dict(_N, loss_best_of_k_tau=0.001)
F_ARMS['f_ab_kmean'] = dict(_N, loss_best_of_k_tau=100.0)
# Sinkhorn entropy and iterations (Joao, 2026-11-25), FOCUS-OT is eps 0.1 / 60 iterations. Measured on demo windows
# (cosine cost, T = 32): eps 0.1 / 60 it. is converged (row-marginal error 2e-2 Hopper, 1e-6 Humanoid); eps 0.03 / 60 it.
# is not (0.19 / 0.03), so eps 0.03 is also run at 200 iterations, whose one-setting control is f_ab_iters200.
for _eps in (0.03, 0.3, 1.0):
    F_ARMS[f'f_ab_eps{_eps:g}'] = dict(_N, loss_ot_eps=_eps)
for _it in (20, 200):
    F_ARMS[f'f_ab_iters{_it}'] = dict(_N, loss_ot_iters=_it)
F_ARMS['f_ab_eps0.03_iters200'] = dict(_N, loss_ot_eps=0.03, loss_ot_iters=200)
# ordering test (p5order): positions-only matching; f_main_ot (all features) is the control
F_ARMS['f_pos_ot'] = dict(_N, _positions=True)
F_ARMS['f_pos_dtw'] = dict(_N, _positions=True, imitation_loss_type='softdtw')
F_ARMS['f_pos_noalign'] = dict(_N, _positions=True, imitation_loss_type='index')
F_ARMS['f_pos_k1'] = dict(_N, _positions=True, loss_best_of_k_k=1)
# 4. zeroth-order FOCUS (batch p4zo, control f_main_ot): the same objective and reward with PPO / SAC
F_ARMS['f_zo_ppo'] = dict(_N, _agent='FOCUS/DFlexAntPPO')
F_ARMS['f_zo_sac'] = dict(_N, _agent='FOCUS/DFlexAntSAC')
# light tuning of the score-function arm (batch p4sftune, tuning seeds): actor learning rate (default 2e-3)
for lr in (2e-3, 6e-4, 2e-4, 6e-5, 2e-5):
    F_ARMS[f'f_sf_lr{lr:g}'] = dict(_N, actor_grad_estimator='score_function',
                                    _shac=dict(actor_optim_kwargs=dict(lr=lr, betas=[0.7, 0.95])))
for rho in (0.03, 0.1, 0.3, 1.0, 3.0):
    F_ARMS[f'f_uot_expert_rho{rho:g}'] = dict(loss_ot_marginal_rho=rho, loss_ot_marginal_side='expert')
    F_ARMS[f'f_uot_both_rho{rho:g}'] = dict(loss_ot_marginal_rho=rho, loss_ot_marginal_side='both')
ARMS.update({k: v for k, v in F_ARMS.items()})

# Run-level settings, the same for every arm
EVAL_EVERY = 500_000
# Timing batch: a fraction of the budget on one GPU model; full-run times are extrapolated from it (REPORT)
BUDGET_SCALE = {'p3time': 0.05}


def budget_of(batch, task, arm):
    b = OOPS_BUDGET if arm.startswith('bl_oops') else {**TASKS, **REWARPED_TASKS}[task][1]
    return int(b * BUDGET_SCALE.get(batch, 1.0))


ZO_AGENTS = {'FOCUS/DFlexAntPPO': 'ppo', 'FOCUS/DFlexAntSAC': 'sac'}
FIRST_ORDER_ONLY = dict(actor_grad_estimator='pathwise', critic_disabled=False, backbone_entropy='none')


def task_file(batch, task, arm):
    demo, _ = {**TASKS, **REWARPED_TASKS}[task]
    budget = budget_of(batch, task, arm)
    base = REFERENCE if arm == 'ref_otcos' else FINAL if arm.startswith('f_') else DEFAULT
    # every F_ARMS entry must get FINAL (the p0screen8 'fn_' arms are the one recorded exception, see above)
    if arm in F_ARMS and base is not FINAL and arm not in FN_ON_DEFAULT_AS_RAN:
        raise ValueError(f'{arm}: F_ARMS entry would not use the FINAL base; name it f_...')
    focus = copy.deepcopy(base)
    if arm == 'ref_otcos' and task == 'ant':
        focus['input_type'] = 'state_state'
    change = dict(ARMS[arm])
    n_trajs = change.pop('_n_trajs', 8)
    agent_name = change.pop('_agent', 'FOCUS/DFlexAntSAPO')
    shac_extra = change.pop('_shac', {})
    if change.pop('_positions', False):
        change['loss_feature_dims'] = POSITION_DIMS[task]
    focus.update(change)
    zo = ZO_AGENTS.get(agent_name)
    if zo:
        # zeroth-order FOCUS: same objective and reward (reward.build_focus_loss), PPO / SAC actor-critic. The
        # first-order-only settings must hold FOCUS's values here and are then dropped (not keys of these agents).
        for k, v in FIRST_ORDER_ONLY.items():
            if focus.pop(k, v) != v:
                raise ValueError(f'{arm}: {k} is a first-order setting and cannot change in a zeroth-order arm')
        if shac_extra:
            raise ValueError(f'{arm}: _shac settings do not apply to {agent_name}')
    focus = dict(demos=dict(path='experts/demos/' + demo, n_trajs=n_trajs), **focus)
    agent = dict(
        name=agent_name,
        ckpt_every=-1,  # no periodic checkpoints: final.pth (= at budget) only
        **RESUME,
        protocol_eval=True,  # shared evaluation of every method (Agent.protocol_*), on a dedicated env
        eval_every_steps=EVAL_EVERY,
        save_eval_rollouts=False,
    )
    if zo:  # their own networks (actor 128-64-32, as FOCUS); save_eval_rollouts is a SHAC-family key
        agent.pop('save_eval_rollouts')
        agent.update({zo: dict(max_agent_steps=budget), 'focus': focus})
    else:  # key order as in every earlier task file
        agent['shac'] = dict(max_epochs=50000, max_agent_steps=budget, **shac_extra)
        agent['focus'] = focus
        agent['network'] = dict(actor_kwargs=dict(mlp_kwargs=dict(units=[128, 64, 32])),
                                critic_kwargs=dict(mlp_kwargs=dict(units=[64, 64])))
    if task in REWARPED_TASKS:
        if zo or agent_name != 'FOCUS/DFlexAntSAPO':
            raise ValueError(f'{arm}: only the default FOCUS agent is set up for {task}')
        agent['name'] = 'FOCUS/RewarpedJumperSAPO'
        agent['network'] = copy.deepcopy(JUMPER_AGENT['network'])
        agent['shac'].update(copy.deepcopy(JUMPER_AGENT['shac']))
        cfg = dict(task=dict(name='Rewarped', env=dict(env_name='Jumper', env_suite='gradsim')), agent=agent,
                   num_envs=32)
    else:
        cfg = dict(task=dict(name='DFlex', env=dict(env_name=task)), agent=agent)
    return dict(
        **cfg,
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


# OOPS cannot reach 10 / 15 M steps in 12 h (0.66 M steps/h): reported at this budget, marked with a dagger (Joao)
OOPS_BUDGET = 4_000_000


def baseline_task_file(batch, task, arm):
    _, rest = arm.split('_', 1)
    method, _, variant = rest.partition('_')
    if variant == 'tuned':  # the variant frozen per task from the tuning batch
        variant = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   'tuned_variants.yaml')))[method][task]
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
    budget = budget_of(batch, task, arm)
    ag = cfg['agent']
    if fam is not None:
        ag.setdefault(fam, {})['max_agent_steps'] = budget
    ag['ckpt_every'] = -1
    ag.update(RESUME)
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


def _replace_demo_file(obj, demo, noisy):
    if isinstance(obj, dict):
        return {k: _replace_demo_file(v, demo, noisy) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_demo_file(v, demo, noisy) for v in obj]
    if isinstance(obj, str) and obj.endswith(demo):
        return obj[: -len(demo)] + noisy
    return obj


def anneal_task_file(batch, task, arm):
    """<arm>__anneal: a SHAC-family arm (FOCUS, ILD) whose linear learning-rate schedule spans the run (max_epochs =
    the epochs of the step budget, 64 envs x 32 steps each), so the rate reaches min_lr at the end. As run, max_epochs is
    50000 (a cap) and the rate ends at 85-96 % of its start (Joao's question on late collapses, 2026-11-26)."""
    base = arm[: -len('__anneal')]
    body = (baseline_task_file if base.startswith('bl_') else task_file)(batch, task, base)
    shac = body['agent'].get('shac')
    if not shac or 'max_epochs' not in shac:
        raise ValueError(f'{arm}: not a SHAC-family arm with max_epochs')
    per_epoch = 64 * int(shac.get('horizon_len', 32))
    shac['max_epochs'] = -(-int(shac['max_agent_steps']) // per_epoch)
    body['logdir'] = f'workdir/focus-{batch}-{task}-{arm}/$(date +%Y%m%d-%H%M%S.%2N)'
    return body


def noisy_task_file(batch, task, arm):
    """<arm>__n<sigma>: the arm with its demonstrations replaced by the noisy copy (scripts/focus/make_noisy_demos.py:
    Gaussian noise of sigma x the per-dimension std on every observation, fixed seed, first 8 episodes). The file is
    shipped with the run (launch.py --attach_input experts/demos=shared/attach/noisy_demos/<task>_<sigma>)."""
    base, sigma = arm.split('__n')
    body = (baseline_task_file if base.startswith('bl_') else task_file)(batch, task, base)
    demo = {**TASKS, **REWARPED_TASKS}[task][0]
    noisy = demo[:-3] + f'_noise{float(sigma):g}.pt'
    out = _replace_demo_file(body, demo, noisy)
    if out == body:
        raise ValueError(f'{arm}: demonstration file {demo} not found in the task file')
    out['logdir'] = f'workdir/focus-{batch}-{task}-{arm}/$(date +%Y%m%d-%H%M%S.%2N)'
    return out


def _set_n_trajs(obj, n):
    """Set n_trajs in every demonstration block ({path: ..., n_trajs: ...}) of a task file; returns the count."""
    count = 0
    if isinstance(obj, dict):
        if 'n_trajs' in obj and str(obj.get('path', '')).endswith('.pt'):
            obj['n_trajs'] = n
            count += 1
        for v in obj.values():
            count += _set_n_trajs(v, n)
    elif isinstance(obj, list):
        for v in obj:
            count += _set_n_trajs(v, n)
    return count


def demos_task_file(batch, task, arm):
    """<arm>__d<N>: a baseline with N demonstrations instead of 8 (the first N episodes of the pool, as for the FOCUS
    arms f_ab_demos<N>); everything else as the arm (few-demonstration study, writer request R16, 2026-11-25)."""
    base, n = arm.split('__d')
    if not base.startswith('bl_'):
        raise ValueError(f'{arm}: FOCUS demonstration counts are the f_ab_demos<N> arms')
    body = baseline_task_file(batch, task, base)
    if _set_n_trajs(body, int(n)) == 0:
        raise ValueError(f'{arm}: no demonstration block in the task file')
    body['logdir'] = f'workdir/focus-{batch}-{task}-{arm}/$(date +%Y%m%d-%H%M%S.%2N)'
    return body


# ckpt/resume.pth at the end of training (step budget or job clock), to extend a run or continue a stopped one exactly
# with resume_from=<file> (Joao, 2026-11-26). Results are unchanged by it (scripts/resume_check.py). '+': the key is not
# in the agent configs.
RESUME = {'+resume_ckpt': True}


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
            if arm.endswith('__anneal'):
                body = anneal_task_file(batch, task, arm)
            elif '__n' in arm:
                body = noisy_task_file(batch, task, arm)
            elif '__d' in arm:
                body = demos_task_file(batch, task, arm)
            else:
                body = baseline_task_file(batch, task, arm) if arm.startswith('bl_') else task_file(batch, task, arm)
            hdr = (f'# Generated by scripts/focus/make_task_files.py ({batch}, arm {arm}); do not edit.\n'
                   f'# Settings are in the body, not in the sweep block (--set seed replaces the sweep block).\n')
            with open(path, 'w') as f:
                f.write(hdr + yaml.safe_dump(body, sort_keys=False, default_flow_style=None))
    print(f'{len(tasks) * len(arms)} task files in {out}')


if __name__ == '__main__':
    main()
