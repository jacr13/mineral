import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nets import MLP, Dist, MultiEncoder
from .utils import weight_init_, weight_init_orthogonal_, weight_init_uniform_


class Actor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        tanh_policy=True,
        fixed_sigma=None,
        mlp_kwargs=None,
        dist_kwargs=None,
        weight_init=None,
    ):
        if mlp_kwargs is None:
            mlp_kwargs = {"units": [512, 256, 128], "act_type": "ELU"}
        if dist_kwargs is None:
            dist_kwargs = {}
        super().__init__()
        self.tanh_policy = tanh_policy
        self.fixed_sigma = fixed_sigma

        self.actor_mlp = MLP(state_dim, **mlp_kwargs)
        self.mu = nn.Linear(self.actor_mlp.out_dim, action_dim)
        if self.tanh_policy:
            pass
        else:
            if self.fixed_sigma is None:
                pass
            elif self.fixed_sigma:
                self.sigma = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32), requires_grad=True)
            else:
                self.sigma = nn.Linear(self.actor_mlp.out_dim, action_dim)
            self.dist = Dist(**dist_kwargs)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init is None:
            pass
        elif self.weight_init == "orthogonal":  # drqv2
            self.apply(weight_init_orthogonal_)
        elif self.weight_init == "uniform":  # original DDPG paper
            self.apply(weight_init_uniform_)
            nn.init.uniform_(self.mu.weight, -0.003, 0.003)
        else:
            raise NotImplementedError(self.weight_init)

    def forward(self, x, std=None):
        if isinstance(x, dict):
            x = x["z"]
        x = self.actor_mlp(x)
        mu = self.mu(x)
        if self.tanh_policy:  # DDPG
            mu = mu.tanh()
            sigma, distr = None, None
        else:  # SAC
            if self.fixed_sigma is None:
                assert std is not None
                sigma = std
            elif self.fixed_sigma:
                sigma = self.sigma
            else:
                sigma = self.sigma(x)
            mu, sigma, distr = self.dist(mu, sigma)
        return mu, sigma, distr


class InverseModel(nn.Module):
    def __init__(
        self,
        obs_space=None,
        action_dim=None,
        input_dim=None,
        mlp_kwargs=None,
        encoder_kwargs=None,
        weight_init=None,
    ):
        if mlp_kwargs is None:
            mlp_kwargs = {"units": [256, 256], "norm_type": "LayerNorm", "act_type": "SiLU"}
        super().__init__()
        self.obs_space = obs_space
        self.weight_init = weight_init

        if obs_space is not None:
            if action_dim is None:
                raise ValueError("action_dim must be set when using obs_space.")
            if "obs" in obs_space:
                self.encoder = nn.Identity()
                mlp_in_dim = obs_space["obs"][0]
            else:
                if encoder_kwargs is None:
                    encoder_kwargs = {}
                self.encoder = MultiEncoder(obs_space, encoder_kwargs, weight_init_fn=weight_init_)
                mlp_in_dim = self.encoder.out_dim
            mlp_in_dim *= 2
        else:
            if input_dim is None or action_dim is None:
                raise ValueError("input_dim and action_dim must be set when obs_space is None.")
            self.encoder = None
            mlp_in_dim = input_dim

        self.mlp = MLP(mlp_in_dim, out_dim=action_dim, plain_last=True, **mlp_kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init is not None:
            weight_init_(self.mlp, self.weight_init)

    def _encode(self, obs_dict):
        if "obs" in self.obs_space:
            z = obs_dict["obs"]
        else:
            encoder_out = self.encoder(obs_dict)
            z = encoder_out["z"]
        return z

    def forward(self, obs, next_obs=None):
        if self.encoder is None:
            if isinstance(obs, dict):
                obs = obs["z"]
            x = obs
        else:
            if next_obs is None:
                raise ValueError("next_obs must be provided when using obs_space.")
            z1 = self._encode(obs)
            z2 = self._encode(next_obs)
            x = torch.cat([z1, z2], dim=-1)
        return self.mlp(x)

    def _normalize_obs(self, obs, normalizer):
        if normalizer is None:
            return obs
        if isinstance(obs, dict) and hasattr(normalizer, "__getitem__"):
            return {k: normalizer[k].normalize(v) for k, v in obs.items()}
        if hasattr(normalizer, "normalize"):
            return normalizer.normalize(obs)
        if callable(normalizer):
            return normalizer(obs)
        return obs

    def train_batch(self, obs, actions, next_obs, optimizer, normalizer=None):
        obs = self._normalize_obs(obs, normalizer)
        next_obs = self._normalize_obs(next_obs, normalizer)
        pred_actions = self(obs, next_obs)
        loss = F.mse_loss(pred_actions, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach()

    def train_on_replay(
        self,
        replay_buffer,
        optimizer,
        batch_size,
        device="cuda",
        iters=1,
        normalizer=None,
        patience=0,
        min_delta=0.0,
    ):
        if replay_buffer.cur_capacity < batch_size:
            return None
        iters = int(iters)
        if iters < 1:
            return None
        patience = 0 if patience is None else int(patience)
        min_delta = float(min_delta)
        self.train()
        losses = []
        best_loss = None
        bad_count = 0
        for _ in range(iters):
            obs, actions, _, next_obs, _ = replay_buffer.sample_batch(batch_size, device=device)
            loss = self.train_batch(obs, actions, next_obs, optimizer, normalizer=normalizer)
            losses.append(loss)
            if patience > 0:
                loss_val = float(loss.item())
                if best_loss is None or (best_loss - loss_val) > min_delta:
                    best_loss = loss_val
                    bad_count = 0
                else:
                    bad_count += 1
                    if bad_count >= patience:
                        break
        self.eval()
        return torch.stack(losses).mean().detach()


class EnsembleQ(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        n_critics=2,
        mlp_kwargs=None,
        weight_init=None,
    ):
        if mlp_kwargs is None:
            mlp_kwargs = {"units": [512, 256, 128], "act_type": "ELU"}
        super().__init__()
        self.n_critics = n_critics
        critics = []
        for _ in range(n_critics):
            q = MLP(state_dim + action_dim, out_dim=1, plain_last=True, **mlp_kwargs)
            critics.append(q)
        self.critics = nn.ModuleList(critics)

        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        for critic in self.critics:
            if self.weight_init is None:
                pass
            elif self.weight_init == "orthogonal":  # drqv2
                critic.apply(weight_init_orthogonal_)
            elif self.weight_init == "uniform":  # original DDPG paper
                critic.apply(weight_init_uniform_)
                nn.init.uniform_(critic.mlp[-1].weight, -0.003, 0.003)
            else:
                raise NotImplementedError(self.weight_init)

    def forward(self, state, action):
        if isinstance(state, dict):
            state = state["z"]
        input_x = torch.cat((state, action), dim=1)
        Qs = [critic(input_x) for critic in self.critics]
        return Qs

    def get_q_min(self, state, action):
        Qs = self.forward(state, action)
        return torch.min(torch.stack(Qs), dim=0).values

    def get_q_values(self, state, action):
        return self.forward(state, action)


class DistributionalEnsembleQ(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        v_min=-10,
        v_max=10,
        num_atoms=51,
        n_critics=2,
        mlp_kwargs=None,
        weight_init=None,
    ):
        if mlp_kwargs is None:
            mlp_kwargs = {"units": [512, 256, 128], "act_type": "ELU"}
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.z_atoms = torch.linspace(v_min, v_max, num_atoms)

        self.n_critics = n_critics
        critics = []
        for _ in range(n_critics):
            q = MLP(state_dim + action_dim, out_dim=num_atoms, plain_last=True, **mlp_kwargs)
            critics.append(q)
        self.critics = nn.ModuleList(critics)

        self.weight_init = weight_init
        self.reset_parameters()

    @property
    def distl(self):
        return True

    def reset_parameters(self):
        if self.weight_init is not None:
            raise NotImplementedError(self.weight_init)

    def forward(self, state, action):
        if isinstance(state, dict):
            state = state["z"]
        input_x = torch.cat((state, action), dim=1)
        Qs = [critic(input_x) for critic in self.critics]
        return Qs

    def get_q_min(self, state, action):
        Qs = self.get_q_values(state, action)
        Qs = [torch.sum(Q * self.z_atoms.to(Q.device), dim=1) for Q in Qs]
        return torch.min(torch.stack(Qs), dim=0).values

    def get_q_values(self, state, action):
        Qs = self.forward(state, action)
        Qs = [torch.softmax(Q, dim=1) for Q in Qs]
        return Qs
