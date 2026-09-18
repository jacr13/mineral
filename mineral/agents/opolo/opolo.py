import torch
import torch.nn.functional as F

from ..dac.dac import DAC


class OPOLO(DAC):
    """Off-Policy Learning from Observations (OPOLO), built on this repo's DAC.

    Reference: Zhu, Lin, Dai, Zhou, "Off-Policy Imitation Learning from
    Observations" (NeurIPS 2020), https://arxiv.org/abs/2102.13185
    Official code: https://github.com/illidanlab/opolo-code
    (see ``opolo-baselines/stable_baselines/td3/td3opolo.py``)
    """

    def __init__(self, full_cfg, **kwargs):
        super().__init__(full_cfg, **kwargs)
        self.opolo_config = full_cfg.agent.get("opolo", {})
        # Matches td3opolo.py's own default (`alpha=0.1`).
        self.opolo_alpha = float(self.opolo_config.get("alpha", 0.1))

    def update_critic(self, obs, action, reward, next_obs, done):
        with torch.no_grad():
            next_obs_n = self._normalize_obs_dict(next_obs)
            next_z = self.encoder_target(next_obs_n)
            next_actions, _, log_prob = self.get_actions(z=next_z, logprob=True)
            target_Q = self.critic_target.get_q_min(next_z, next_actions)
            if self.sac_config.backup_entropy:
                target_Q -= self.get_alpha() * log_prob
            target_Q = reward + (1 - done) * (self.sac_config.gamma**self.sac_config.nstep) * target_Q

        obs_n = self._normalize_obs_dict(obs)
        z = self.encoder(obs_n)
        current_Qs = self.critic.get_q_values(z, action)
        critic_loss = torch.sum(torch.stack([F.mse_loss(current_Q, target_Q) for current_Q in current_Qs]))

        # OPOLO / AlgaeDICE correction: push the critic's value of the CURRENT
        # policy's action down at the (replay-approximated) initial state
        # distribution, scaled by 2*alpha*(1-gamma). The actor already
        # maximizes this exact quantity via its own SAC objective
        # (Q(s, pi(s)), in `DAC.update_actor`/SAC's actor loss); minimizing it
        # here on the critic side is what creates the adversarial dynamic.
        # `pi_actions` is sampled under no_grad so this term only trains the
        # critic (and its encoder), matching the reference's var_list scoping.
        with torch.no_grad():
            pi_actions = self.get_actions(z=z, sample=True)
        qf_init = self.critic.get_q_min(z, pi_actions).mean()
        dice_term = 2.0 * self.opolo_alpha * (1.0 - self.sac_config.gamma) * qf_init
        critic_loss = critic_loss + dice_term

        grad_norm = self.optimizer_update(self.critic_optim, critic_loss)

        target_values = {
            "target_values/mean": target_Q.mean(),
            "target_values/std": target_Q.std(),
            "target_values/max": target_Q.max(),
            "target_values/min": target_Q.min(),
            "opolo/qf_init": qf_init.detach(),
            "opolo/dice_term": dice_term.detach(),
        }
        return critic_loss, grad_norm, target_values
