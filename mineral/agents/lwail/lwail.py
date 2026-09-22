import torch

from ..dac.dac import DAC
from ..gail.models import Discriminator
from .models import PhiNet


class LWAIL(DAC):
    """Latent Wasserstein Adversarial Imitation Learning, built on this repo's DAC.

    Reference: Yang, Yan, Schwing, Wang, "Latent Wasserstein Adversarial
    Imitation Learning" (ICLR 2026), https://arxiv.org/abs/2603.05440
    Official code: https://github.com/JackyYang258/LWAIL (see ``core.py``,
    ``network.py``, ``utils.py``).

    LWAIL swaps DAC/GAIL's binary-classification discriminator for a
    1-Lipschitz Wasserstein critic ("f_net" upstream): trained with a
    tanh-bounded IPM objective plus a WGAN-GP gradient penalty
    (``upstream: core.py::Agent.f_update/pretrain``, ``utils.py::gradient_penalty``),
    instead of BCE, and it shapes reward as ``sigmoid(-f_net(...))`` instead of
    ``-log(1 - sigmoid(f_net(...)))``. Everything else (demos, off-policy
    actor-critic, replay buffer, ``state``/``state_action``/``state_state``
    discriminator inputs, inverse-model regularization) is inherited unchanged
    from :class:`~mineral.agents.dac.dac.DAC`.

    Two upstream options are ported as-is:
      - ``lwail.minus``: for ``input_type: state_state``, feed the critic
        ``next_obs - obs`` instead of raw ``next_obs`` (upstream ``--minus``).
      - ``lwail.using_icvf``: embed states through a frozen, pretrained ICVF
        encoder (:class:`~mineral.agents.lwail.models.PhiNet`) before feeding
        the critic (upstream ``--using_icvf``). Upstream only ships ICVF
        checkpoints for a few D4RL Mujoco tasks (dimension-incompatible with
        this repo's dflex tasks) and treats ICVF pretraining itself as an
        external/TODO codebase, so this defaults to off; enabling it requires
        supplying a compatible checkpoint via ``lwail.icvf_path``.

    Upstream also pretrains "f_net" for ``lwail.pretrain_iters`` steps (default
    2500, matching upstream's hardcoded value) against the warm-up random
    rollout before online training starts (``upstream: core.py::Agent.pretrain``);
    this is ported via DAC's ``_post_warmup`` hook.
    """

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)

        self.lwail_config = full_cfg.agent.get("lwail", {})
        self.gp_coef = float(self.lwail_config.get("gp_coef", 10.0))  # upstream `--alpha`
        self.minus = bool(self.lwail_config.get("minus", True))
        self.pretrain_iters = int(self.lwail_config.get("pretrain_iters", 2500))

        self.using_icvf = bool(self.lwail_config.get("using_icvf", False))
        self.phi_net = None
        if self.using_icvf:
            icvf_path = self.lwail_config.get("icvf_path", None)
            assert icvf_path, "lwail.icvf_path must be set when lwail.using_icvf=true"
            icvf_hidden_dims = list(self.lwail_config.get("icvf_hidden_dims", [256, 256]))

            obs_dim = self.obs_space["obs"][0]
            self.phi_net = PhiNet([obs_dim] + icvf_hidden_dims).to(self.device)
            self.phi_net.load_state_dict(torch.load(icvf_path, map_location=self.device, weights_only=False))
            self.phi_net.eval()
            for p in self.phi_net.parameters():
                p.requires_grad_(False)

            # Rebuild the discriminator (constructed by DAC.__init__ against the
            # raw obs space) to instead operate on the ICVF-embedded state space.
            embed_dim = icvf_hidden_dims[-1]
            disc_obs_space = {"obs": (embed_dim,)}
            discriminator_config = self.dac_config.get("discriminator", {})
            encoder_kwargs = discriminator_config.get("encoder_kwargs", None)
            if encoder_kwargs is None:
                encoder_kwargs = self.network_config.get("encoder_kwargs", {})
            self.discriminator = Discriminator(
                disc_obs_space,
                self.action_dim,
                input_type=self.input_type,
                discriminator_kwargs=discriminator_config,
                encoder_kwargs=encoder_kwargs,
            ).to(self.device)
            disc_optim_kwargs = (
                self.dac_config.get("discriminator_optim")
                or discriminator_config.get("optim")
                or {"type": "Adam", "kwargs": {"lr": 3e-4}}
            )
            DiscOptim = getattr(torch.optim, disc_optim_kwargs.get("type", "Adam"))
            self.disc_optim = DiscOptim(self.discriminator.parameters(), **disc_optim_kwargs.get("kwargs", {}))
            self.discriminator.eval()

    def get_discriminator_inputs(self, obs, actions, next_obs):
        if self.using_icvf:
            obs = {"obs": self.phi_net(obs["obs"])}
            next_obs = {"obs": self.phi_net(next_obs["obs"])}

        if self.input_type == "state":
            input_1, input_2 = obs, None
        elif self.input_type == "state_action":
            input_1, input_2 = obs, actions
        elif self.input_type == "state_state":
            next_obs_in = next_obs
            if self.minus:
                next_obs_in = {"obs": next_obs["obs"] - obs["obs"]}
            input_1, input_2 = obs, next_obs_in
        else:
            raise NotImplementedError

        return input_1, input_2

    @torch.no_grad()
    def dac_reward(self, obs, actions, next_obs):
        # LWAIL reward: sigmoid(-f_net(...)), vs. DAC/GAIL's -log(1 - sigmoid(f_net(...))).
        input_1, input_2 = self.get_discriminator_inputs(obs, actions, next_obs)
        logits = self.discriminator(input_1, input_2)
        reward = torch.sigmoid(-logits)
        return (reward * self.reward_scale).unsqueeze(-1)

    @staticmethod
    def _leaf_tensors(x):
        if x is None:
            return []
        if isinstance(x, dict):
            return list(x.values())
        return [x]

    @staticmethod
    def _interp(a, b, eps):
        if a is None:
            return None
        if isinstance(a, dict):
            out = {}
            for k in a:
                v = (eps * a[k] + (1.0 - eps) * b[k]).detach()
                v.requires_grad_(True)
                out[k] = v
            return out
        v = (eps * a + (1.0 - eps) * b).detach()
        v.requires_grad_(True)
        return v

    def _gradient_penalty(self, exp_input_1, exp_input_2, pol_input_1, pol_input_2):
        # Ported from upstream `utils.py::gradient_penalty`, adapted to the
        # dict/tensor discriminator inputs used throughout this repo: interpolate
        # each raw component separately with a shared per-sample coefficient
        # (equivalent to interpolating the concatenated vector upstream does),
        # then penalize the critic's input gradient norm for deviating from 1.
        batch_size = exp_input_1["obs"].shape[0]
        eps = torch.rand(batch_size, 1, device=self.device)

        interp_1 = self._interp(exp_input_1, pol_input_1, eps)
        interp_2 = self._interp(exp_input_2, pol_input_2, eps)

        logits = self.discriminator(interp_1, interp_2)
        inputs = self._leaf_tensors(interp_1) + self._leaf_tensors(interp_2)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=inputs,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        grad_sq = sum((g.reshape(g.shape[0], -1) ** 2).sum(dim=1) for g in grads)
        grad_norm = torch.sqrt(grad_sq + 1e-12)
        return ((grad_norm - 1.0) ** 2).mean() * self.gp_coef

    def _disc_step(self, exp_input_1, exp_input_2, pol_input_1, pol_input_2):
        # Ported from upstream `core.py::Agent.f_update`'s inner loop: a tanh-bounded
        # IPM (mean critic value on expert vs. policy data) plus a WGAN-GP penalty.
        exp_logits = self.discriminator(exp_input_1, exp_input_2)
        pol_logits = self.discriminator(pol_input_1, pol_input_2)
        ipm = torch.tanh(exp_logits).mean() - torch.tanh(pol_logits).mean()
        gp = self._gradient_penalty(exp_input_1, exp_input_2, pol_input_1, pol_input_2)
        loss = ipm + gp

        self.disc_optim.zero_grad()
        loss.backward()
        self.disc_optim.step()
        return loss.detach(), ipm.detach(), gp.detach()

    def train_discriminator(self, memory):
        if memory.cur_capacity < self.policy_batch_size:
            return {"discriminator/total": 0.0, "discriminator/ipm": 0.0, "discriminator/gp": 0.0}

        self.discriminator.train()

        # Matches upstream: both the policy and expert batches are sampled once
        # per call, then `disc_iters` (upstream `f_epoch`) gradient steps are
        # taken against that same fixed pair (only the GP interpolation is
        # re-randomized every step).
        pol_obs, pol_act, _, pol_next_obs, _ = memory.sample_batch(self.policy_batch_size, device=self.device)
        if self.normalize_input:
            pol_obs = {k: self.obs_rms[k].normalize(v) for k, v in pol_obs.items()}
            pol_next_obs = {k: self.obs_rms[k].normalize(v) for k, v in pol_next_obs.items()}
        exp_obs, exp_act, exp_next_obs = self._sample_batch(
            self.demos["obs"],
            self.demos["act"],
            self.expert_batch_size,
            dones=self.demos["done"],
            next_obs_rollout=self.demos["next_obs"],
        )

        pol_input_1, pol_input_2 = self.get_discriminator_inputs(pol_obs, pol_act, pol_next_obs)
        exp_input_1, exp_input_2 = self.get_discriminator_inputs(exp_obs, exp_act, exp_next_obs)

        losses, ipms, gps = [], [], []
        for _ in range(self.disc_iters):
            loss, ipm, gp = self._disc_step(exp_input_1, exp_input_2, pol_input_1, pol_input_2)
            losses.append(loss)
            ipms.append(ipm)
            gps.append(gp)

        self.discriminator.eval()
        return {
            "discriminator/total": torch.stack(losses).mean().item(),
            "discriminator/ipm": torch.stack(ipms).mean().item(),
            "discriminator/gp": torch.stack(gps).mean().item(),
        }

    def _post_warmup(self):
        # Ported from upstream `core.py::Agent.pretrain`: warm-start the critic
        # against the warm-up random rollout before online training starts. The
        # policy-side batch is sampled once and held fixed (as upstream does),
        # while the expert batch is refreshed every iteration (also as upstream
        # does, unlike the periodic `train_discriminator` updates above).
        if self.pretrain_iters <= 0 or self.memory.cur_capacity < self.policy_batch_size:
            return

        print(f"[LWAIL] Pretraining discriminator for {self.pretrain_iters} iters on warm-up rollout vs. expert demos...")
        self.discriminator.train()

        pol_obs, pol_act, _, pol_next_obs, _ = self.memory.sample_batch(self.policy_batch_size, device=self.device)
        if self.normalize_input:
            pol_obs = {k: self.obs_rms[k].normalize(v) for k, v in pol_obs.items()}
            pol_next_obs = {k: self.obs_rms[k].normalize(v) for k, v in pol_next_obs.items()}
        pol_input_1, pol_input_2 = self.get_discriminator_inputs(pol_obs, pol_act, pol_next_obs)

        losses = []
        for _ in range(self.pretrain_iters):
            exp_obs, exp_act, exp_next_obs = self._sample_batch(
                self.demos["obs"],
                self.demos["act"],
                self.expert_batch_size,
                dones=self.demos["done"],
                next_obs_rollout=self.demos["next_obs"],
            )
            exp_input_1, exp_input_2 = self.get_discriminator_inputs(exp_obs, exp_act, exp_next_obs)
            loss, _, _ = self._disc_step(exp_input_1, exp_input_2, pol_input_1, pol_input_2)
            losses.append(loss)

        self.discriminator.eval()
        metrics = {"train_stats/pretrain/discriminator_loss": torch.stack(losses).mean().item()}
        self.writer.add(self.agent_steps, metrics)
        self.writer.write()
