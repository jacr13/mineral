"""Intention-Conditioned Value Functions (ICVF), ported to PyTorch.

LWAIL upstream (https://github.com/JackyYang258/LWAIL) loads a pretrained
ICVF encoder (its ``PhiNet``, see ``models.py``) for ``using_icvf: true``, but
only ships checkpoints for a few D4RL Mujoco tasks and points to a separate,
unfinished "ICVF-PyTorch Repository (todo)" for the actual training code.
This module is that missing training code: a from-scratch PyTorch port of the
official JAX implementation, https://github.com/dibyaghosh/icvf_release (see
``src/icvf_learner.py``, ``src/icvf_networks.py``, ``src/gc_dataset.py``),
reproducing the method from Ghosh et al., "Reinforcement Learning from
Passive Data via Latent Intentions" (ICML 2023).

ICVF learns a 3-argument value function V(s, g, z) -- "how close is s to
outcome g, if the agent's intention is to reach z" -- from reward-free,
state-only trajectories via hindsight goal relabeling (as in goal-conditioned
RL / HER) and an IQL-style expectile regression. It factorizes as a bilinear
form over per-state features phi(s) and per-goal features psi(g), gated by an
intention-dependent transform T(z); phi(s) is what downstream code (here,
:class:`~mineral.agents.lwail.lwail.LWAIL`) uses as the "dynamics-aware
latent space." See ``load_or_pretrain_icvf`` for the end-to-end path
``LWAIL`` uses: load a cached checkpoint if one exists at ``lwail.icvf_path``,
otherwise pretrain one from a random rollout and cache it there.
"""

import os
from copy import deepcopy

import torch
import torch.nn as nn

from ..ddpg.utils import soft_update
from .models import PhiNet


class MultilinearVF(nn.Module):
    """Ported from `icvf_networks.py::MultilinearVF`.

    V(s, g, z) = <A(T(z) * phi(s)), B(T(z) * psi(g))>, a low-rank bilinear
    approximation to a full T(z)-conditioned dxd value matrix.
    """

    def __init__(self, obs_dim, hidden_dims=(256, 256)):
        super().__init__()
        hidden_dims = list(hidden_dims)
        feat_dim = hidden_dims[-1]
        self.phi_net = PhiNet([obs_dim] + hidden_dims)
        self.psi_net = PhiNet([obs_dim] + hidden_dims)
        self.T_net = PhiNet([feat_dim] + hidden_dims)
        self.matrix_a = nn.Linear(feat_dim, feat_dim)
        self.matrix_b = nn.Linear(feat_dim, feat_dim)

    def get_phi(self, obs):
        return self.phi_net(obs)

    def forward(self, s, g, z):
        phi = self.phi_net(s)
        psi = self.psi_net(g)
        Tz = self.T_net(self.psi_net(z))
        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        return (phi_z * psi_z).sum(dim=-1)


class ICVFEnsemble(nn.Module):
    """Two independent `MultilinearVF`s (upstream's `ensemblize(..., 2, ...)`).

    Used for double-Q-style target computation. `get_phi` (used downstream by
    LWAIL) arbitrarily takes member 0 -- unlike the value heads, there's no
    principled way to combine two independently-trained embedding networks'
    weights.
    """

    def __init__(self, obs_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.nets = nn.ModuleList([MultilinearVF(obs_dim, hidden_dims) for _ in range(2)])

    def forward(self, s, g, z):
        return self.nets[0](s, g, z), self.nets[1](s, g, z)

    def get_phi(self, obs, member=0):
        return self.nets[member].get_phi(obs)


def expectile_loss(adv, diff, expectile=0.9):
    weight = torch.where(adv >= 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)


def icvf_loss(value_net, target_value_net, batch, discount=0.99, expectile=0.9, min_q=True, no_intent=False):
    """Ported from `icvf_learner.py::icvf_loss`.

    Two coupled TD errors: (1) how far s is from outcome g (`*_gz`), regressed
    with an expectile that up-weights transitions where the state actually
    progresses toward the intention z (2), an advantage computed the same way
    IQL computes advantages, but for reaching z instead of maximizing reward.
    """
    obs, next_obs = batch["observations"], batch["next_observations"]
    goals, desired_goals = batch["goals"], batch["desired_goals"]
    if no_intent:
        desired_goals = torch.ones_like(desired_goals)

    with torch.no_grad():
        next_v1_gz, next_v2_gz = target_value_net(next_obs, goals, desired_goals)
    q1_gz = batch["rewards"] + discount * batch["masks"] * next_v1_gz
    q2_gz = batch["rewards"] + discount * batch["masks"] * next_v2_gz

    v1_gz, v2_gz = value_net(obs, goals, desired_goals)

    with torch.no_grad():
        next_v1_zz, next_v2_zz = target_value_net(next_obs, desired_goals, desired_goals)
        next_v_zz = torch.minimum(next_v1_zz, next_v2_zz) if min_q else (next_v1_zz + next_v2_zz) / 2
        q_zz = batch["desired_rewards"] + discount * batch["desired_masks"] * next_v_zz

        v1_zz, v2_zz = target_value_net(obs, desired_goals, desired_goals)
        v_zz = (v1_zz + v2_zz) / 2
        adv = q_zz - v_zz
        if no_intent:
            adv = torch.zeros_like(adv)

    value_loss1 = expectile_loss(adv, q1_gz - v1_gz, expectile).mean()
    value_loss2 = expectile_loss(adv, q2_gz - v2_gz, expectile).mean()
    value_loss = value_loss1 + value_loss2

    info = {
        "icvf/value_loss": value_loss.detach(),
        "icvf/v_gz_mean": v1_gz.detach().mean(),
        "icvf/adv_mean": adv.detach().mean(),
        "icvf/accept_prob": (adv.detach() >= 0).float().mean(),
    }
    return value_loss, info


class GCSDataset:
    """Ported from `gc_dataset.py::GCSDataset`.

    Specialized to a dense `(num_trajs, T, obs_dim)` tensor of fixed-length
    random rollouts (rather than upstream's flat, variable-length-episode
    offline dataset), since that's what a vectorized dflex random rollout
    naturally produces.

    Builds (s, s', g, z, reward, mask, desired_reward, desired_mask) batches
    via hindsight relabeling: goals/intents are sampled either uniformly at
    random, from later in the same trajectory (interpolated towards the final
    step), or as the current state itself; `success = (goal == current
    state)` defines a sparse reward, entirely state-only (no env reward or
    actions needed).
    """

    def __init__(
        self,
        observations,
        p_randomgoal=0.3,
        p_trajgoal=0.5,
        p_currgoal=0.2,
        p_samegoal=0.5,
        reward_scale=1.0,
        reward_shift=-1.0,
    ):
        assert abs(p_randomgoal + p_trajgoal + p_currgoal - 1.0) < 1e-6
        self.observations = observations
        self.num_trajs, self.T, self.obs_dim = observations.shape
        self.device = observations.device
        self.p_randomgoal = p_randomgoal
        self.p_trajgoal = p_trajgoal
        self.p_currgoal = p_currgoal
        self.p_samegoal = p_samegoal
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift

    def _sample_goal_indices(self, traj_idx, step_idx):
        batch_size = traj_idx.shape[0]
        device = self.device

        rand_traj = torch.randint(0, self.num_trajs, (batch_size,), device=device)
        rand_step = torch.randint(0, self.T, (batch_size,), device=device)

        final_step = self.T - 1
        distance = torch.rand(batch_size, device=device)
        middle_step = torch.round(step_idx * distance + final_step * (1 - distance)).long().clamp(0, final_step)

        use_traj = torch.rand(batch_size, device=device) < (self.p_trajgoal / (1.0 - self.p_currgoal))
        goal_traj = torch.where(use_traj, traj_idx, rand_traj)
        goal_step = torch.where(use_traj, middle_step, rand_step)

        use_curr = torch.rand(batch_size, device=device) < self.p_currgoal
        goal_traj = torch.where(use_curr, traj_idx, goal_traj)
        goal_step = torch.where(use_curr, step_idx, goal_step)
        return goal_traj, goal_step

    def sample(self, batch_size):
        device = self.device
        traj_idx = torch.randint(0, self.num_trajs, (batch_size,), device=device)
        step_idx = torch.randint(0, self.T - 1, (batch_size,), device=device)

        obs = self.observations[traj_idx, step_idx]
        next_obs = self.observations[traj_idx, step_idx + 1]

        goal_traj, goal_step = self._sample_goal_indices(traj_idx, step_idx)
        desired_traj, desired_step = self._sample_goal_indices(traj_idx, step_idx)
        use_same = torch.rand(batch_size, device=device) < self.p_samegoal
        goal_traj = torch.where(use_same, desired_traj, goal_traj)
        goal_step = torch.where(use_same, desired_step, goal_step)

        success = (traj_idx == goal_traj) & (step_idx == goal_step)
        desired_success = (traj_idx == desired_traj) & (step_idx == desired_step)

        rewards = success.float() * self.reward_scale + self.reward_shift
        desired_rewards = desired_success.float() * self.reward_scale + self.reward_shift
        masks = 1.0 - success.float()
        desired_masks = 1.0 - desired_success.float()

        return {
            "observations": obs,
            "next_observations": next_obs,
            "goals": self.observations[goal_traj, goal_step],
            "desired_goals": self.observations[desired_traj, desired_step],
            "rewards": rewards,
            "masks": masks,
            "desired_rewards": desired_rewards,
            "desired_masks": desired_masks,
        }


def train_icvf(
    observations,
    hidden_dims=(256, 256),
    discount=0.99,
    expectile=0.9,
    target_update_rate=0.005,
    min_q=True,
    no_intent=False,
    p_randomgoal=0.3,
    p_trajgoal=0.5,
    p_currgoal=0.2,
    p_samegoal=0.5,
    reward_scale=1.0,
    reward_shift=-1.0,
    batch_size=256,
    train_steps=20000,
    lr=3e-4,
    log_every=1000,
):
    """Trains an `ICVFEnsemble` on `observations` and returns it.

    `observations` is a `(num_trajs, T, obs_dim)` tensor of state-only
    rollouts, e.g. from `collect_random_rollout`. See `load_or_pretrain_icvf`
    for the end-to-end path `LWAIL` actually uses.
    """
    obs_dim = observations.shape[-1]
    device = observations.device
    dataset = GCSDataset(observations, p_randomgoal, p_trajgoal, p_currgoal, p_samegoal, reward_scale, reward_shift)

    value_net = ICVFEnsemble(obs_dim, hidden_dims).to(device)
    target_value_net = deepcopy(value_net)
    for p in target_value_net.parameters():
        p.requires_grad_(False)

    optim = torch.optim.Adam(value_net.parameters(), lr=lr, eps=1e-8)

    for step in range(1, train_steps + 1):
        batch = dataset.sample(batch_size)
        loss, info = icvf_loss(value_net, target_value_net, batch, discount, expectile, min_q, no_intent)

        optim.zero_grad()
        loss.backward()
        optim.step()
        soft_update(target_value_net, value_net, target_update_rate)

        if log_every > 0 and (step % log_every == 0 or step == train_steps):
            print(
                f"[ICVF] step {step}/{train_steps} "
                f"loss={info['icvf/value_loss'].item():.4f} "
                f"v_gz={info['icvf/v_gz_mean'].item():.4f} "
                f"accept_prob={info['icvf/accept_prob'].item():.3f}"
            )

    return value_net


@torch.no_grad()
def collect_random_rollout(env, num_steps, device):
    """Collects `(num_envs, num_steps, obs_dim)` state-only transitions.

    Under a uniform-random policy, mirroring upstream's
    `core.py::get_random_dataset` and this repo's own warm-up random
    exploration. `env` just needs `num_envs`/`num_obs`/`num_actions`/
    `reset()`/`step()`, matching e.g. the `DDPG`-based agent's own `self.env`.
    """
    obs = env.reset()
    traj = torch.empty((env.num_envs, num_steps, env.num_obs), dtype=torch.float32, device=device)
    for t in range(num_steps):
        traj[:, t] = obs
        actions = torch.rand((env.num_envs, env.num_actions), device=device) * 2.0 - 1.0
        obs, _reward, _done, _info = env.step(actions)
    return traj


def load_or_pretrain_icvf(env, obs_dim, hidden_dims, icvf_path, device, pretrain_kwargs=None):
    """Loads a pretrained ICVF `phi_net` from `icvf_path`.

    Or trains one on a fresh random rollout from `env` and caches it to
    `icvf_path` if not found. This is what makes `lwail.using_icvf: true` a one-command experience: the
    first run for a given `icvf_path` (typically per-environment, since it's
    keyed off `task.env.env_name` by default -- see `LWAIL.__init__`) pays the
    one-time pretraining cost and saves the result; every subsequent run
    (other seeds, sweeps) just loads the cached checkpoint. Returns a frozen,
    eval-mode `PhiNet`.
    """
    if icvf_path and os.path.exists(icvf_path):
        print(f"[LWAIL] Loading pretrained ICVF encoder from {icvf_path}")
        phi_net = PhiNet([obs_dim] + list(hidden_dims)).to(device)
        phi_net.load_state_dict(torch.load(icvf_path, map_location=device, weights_only=False))
    else:
        pretrain_kwargs = pretrain_kwargs or {}
        # Upstream (Table 8 / Sec. 4.2): "10K state pairs collected from random
        # rollouts". Derived from a transition budget (not a fixed rollout_len)
        # so it stays correct regardless of `env.num_envs`.
        num_transitions = pretrain_kwargs.pop("num_transitions", 10000)
        rollout_len = max(1, round(num_transitions / env.num_envs))
        print(
            f"[LWAIL] No ICVF checkpoint at '{icvf_path}'; pretraining one from a "
            f"{env.num_envs}x{rollout_len} (~{env.num_envs * rollout_len}) random rollout..."
        )
        observations = collect_random_rollout(env, rollout_len, device)
        value_net = train_icvf(observations, hidden_dims=hidden_dims, **pretrain_kwargs)
        phi_net = value_net.nets[0].phi_net
        if icvf_path:
            os.makedirs(os.path.dirname(icvf_path) or ".", exist_ok=True)
            torch.save(phi_net.state_dict(), icvf_path)
            print(f"[LWAIL] Saved pretrained ICVF encoder to {icvf_path}")

    phi_net.eval()
    for p in phi_net.parameters():
        p.requires_grad_(False)
    return phi_net
