"""Shared pieces of the imitation agents in this package (non-differentiable-simulator methods)."""

import torch
import torch.nn.functional as F

from ...common.demos import get_demos
from ..ddpg.models import InverseModel


def tree_index(obs, *idx):
    return {k: v[idx] for k, v in obs.items()}


def tree_cat(*obs):
    return {k: torch.cat([o[k] for o in obs]) for k in obs[0]}


def to_device(obs, device):
    return {k: v.to(device) for k, v in obs.items()}


def state_dims(full_cfg, obs_space, enabled=False):
    """Leading coordinates of the flat 'obs' vector that describe the state, or None for the whole vector.

    DFlex Ant / Humanoid / SNU observations end with the previous action, so s' contains a_t. Off by default:
    baselines use the environment's observation as in prior work (FOCUS and the published baselines); with
    ``state_obs: true`` the discriminators, inverse models and matching costs read only the state part, as MENTOR
    does (ILD then fails on Humanoid: its Chamfer cost matched the expert's previous actions).
    """
    if not enabled or 'obs' not in obs_space or full_cfg.task.name != 'DFlex':
        return None
    from ..diffil.mentor_sim import SPECS

    spec = SPECS.get(full_cfg.task.env.env_name)
    if spec is None or spec.obs_dim >= obs_space['obs'][0]:
        return None
    return spec.obs_dim


def slice_obs(obs, d):
    return obs if d is None else {**obs, 'obs': obs['obs'][..., :d]}


def sliced_space(obs_space, d):
    return obs_space if d is None else {**obs_space, 'obs': (d,)}


def finite_rows(*obs):
    """Rows where every tensor of every observation dict is finite."""
    ok = None
    for o in obs:
        for v in o.values():
            f = torch.isfinite(v.reshape(v.shape[0], -1)).all(-1).to(next(iter(obs[0].values())).device)
            ok = f if ok is None else ok & f
    return ok


class ExpertTransitions:
    """Expert (s, a, s') transitions of the demos, with correct episode boundaries.

    A transition (n, t) is valid up to and including the first done of trajectory n, and only if every value is
    finite. A trajectory whose first done comes before its last index terminated (fell): that last transition
    is marked ``terminal`` (its successor is an absorbing state, for DAC / OPOLO). A done at the last index is a
    time limit, not a termination.
    """

    def __init__(self, device, demos_cfg):
        self.device = device
        self.demos = get_demos(device, **demos_cfg)
        obs, next_obs = self.demos['obs'], self.demos['next_obs']
        act, done = self.demos['act'].float(), self.demos['done'].bool()
        N, T = done.shape
        t = torch.arange(T, device=device)[None].expand(N, T)
        has_done = done.any(1)
        first_done = torch.where(has_done, done.float().argmax(1), torch.full((N,), T - 1, device=device))
        valid = t <= first_done[:, None]
        for x in list(obs.values()) + list(next_obs.values()) + [act]:
            valid &= torch.isfinite(x.reshape(N, T, -1)).all(-1)
        terminal = has_done[:, None] & (first_done[:, None] < T - 1) & (t == first_done[:, None])
        n_idx, t_idx = valid.nonzero(as_tuple=True)
        self.obs = tree_index(obs, n_idx, t_idx)
        self.next_obs = tree_index(next_obs, n_idx, t_idx)
        self.act = act[n_idx, t_idx].clamp(-1, 1)
        self.terminal = terminal[n_idx, t_idx]
        self.size = n_idx.shape[0]
        self.expert_return = self.demos['expert_return']
        print(f'Expert transitions: {self.size} from {N} demos ({int(self.terminal.sum())} terminal)')

    def sample(self, batch_size):
        i = torch.randint(0, self.size, (batch_size,), device=self.device)
        return tree_index(self.obs, i), self.act[i], tree_index(self.next_obs, i), self.terminal[i]


def discriminator_loss(exp_logits, pol_logits, label_smooth=0.0):
    """BCE with expert = 1, policy = 0 (optionally smoothed); returns (loss, metrics)."""
    real, fake = 1.0 - label_smooth, label_smooth
    loss_real = F.binary_cross_entropy_with_logits(exp_logits, torch.full_like(exp_logits, real))
    loss_fake = F.binary_cross_entropy_with_logits(pol_logits, torch.full_like(pol_logits, fake))
    with torch.no_grad():
        metrics = {
            'loss_real': loss_real.detach(),
            'loss_fake': loss_fake.detach(),
            'acc_expert': (exp_logits > 0).float().mean(),
            'acc_policy': (pol_logits < 0).float().mean(),
        }
    return loss_real + loss_fake, metrics


def adversarial_reward(logits, kind='gail'):
    """Reward from discriminator logits: gail -log(1 - D) = softplus(l); airl log D - log(1 - D) = l."""
    if kind == 'gail':
        return F.softplus(logits)
    if kind == 'airl':
        return logits
    if kind == 'neg':  # -log D: always negative (no survival bonus)
        return -F.softplus(-logits)
    raise NotImplementedError(kind)


class InverseActionModel:
    """Inverse dynamics a = g(s, s') trained by regression on the agent's own transitions (MAAD, OPOLO, BCO)."""

    def __init__(self, obs_space, action_dim, device, cfg):
        self.net = InverseModel(
            obs_space=obs_space,
            action_dim=action_dim,
            mlp_kwargs=cfg.get('mlp_kwargs', None),
            encoder_kwargs=cfg.get('encoder_kwargs', None) or {},
        ).to(device)
        optim = cfg.get('optim', {'type': 'Adam', 'kwargs': {'lr': 3e-4}})
        self.optim = getattr(torch.optim, optim.get('type', 'Adam'))(self.net.parameters(), **optim.get('kwargs', {}))

    def train_step(self, obs, actions, next_obs):
        self.net.train()
        loss = F.mse_loss(self.net(obs, next_obs), actions.clamp(-1, 1))
        self.optim.zero_grad()
        if torch.isfinite(loss):  # never let one bad batch poison the model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.optim.step()
        self.net.eval()
        return loss.detach()

    @torch.no_grad()
    def __call__(self, obs, next_obs):
        return self.net(obs, next_obs).clamp(-1, 1)

    def state_dict(self):
        return {'net': self.net.state_dict(), 'optim': self.optim.state_dict()}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd['net'])
        self.optim.load_state_dict(sd['optim'])
