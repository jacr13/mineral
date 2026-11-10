import torch.nn as nn


class ChamferImitationLoss(nn.Module):
    """Faithful ILD Chamfer-α loss.

    sim_f: [B, T, d]            learner trajectories
    exp_f: [B_e, T_e, d]        multiple expert trajectories
                                (treated as ONE SINGLE set)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, sim_f, exp_f, **kwargs):
        B, T, d = sim_f.shape
        B_e, T_e, d_e = exp_f.shape
        assert d == d_e

        # Combine all expert trajectories into a single point set
        exp_flat = exp_f.reshape(B_e * T_e, d)  # [M, d]

        # Flatten learner trajectories
        sim_flat = sim_f.reshape(B * T, d)  # [N, d]

        # Pairwise distances between sim and expert
        # Using the identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        sim_sq = (sim_flat**2).sum(dim=1, keepdim=True)
        exp_sq = (exp_flat**2).sum(dim=1).unsqueeze(0)
        d2 = sim_sq + exp_sq - 2 * sim_flat @ exp_flat.T
        d2 = d2.clamp_min(0.0)

        # Deviation term
        min_d2_s_to_e, _ = d2.min(dim=1)  # [N]
        deviation = min_d2_s_to_e.mean()

        # Coverage term
        min_d2_e_to_s, _ = d2.min(dim=0)  # [M]
        coverage = min_d2_e_to_s.mean()

        loss = deviation + self.alpha * coverage
        # similar to the paper code we apply a tanh to the loss
        # https://github.com/sail-sg/ILD/blob/df699447bfe025a3e7e4e2fda5342b49c3942a0d/policy/brax_task/train_on_policy.py#L390
        # https://github.com/sail-sg/ILD/blob/df699447bfe025a3e7e4e2fda5342b49c3942a0d/policy/brax_task/train_multi_traj.py#L367
        # Finally, we don't apply it prevents the model from learning on higher loss values
        # loss = torch.tanh(loss)

        # Per-step deviation reshaped back to [B, T]
        per_step = min_d2_s_to_e.view(B, T)

        info = {
            "loss_deviation": deviation.item(),
            "loss_coverage": coverage.item(),
            "per_step_costs": per_step.detach(),
        }

        return loss, info
