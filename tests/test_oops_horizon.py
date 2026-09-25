"""Regression checks for OOPS's fixed-horizon rollout and TD targets."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mineral.agents.oops.oops import OOPS
from mineral.common import normalizers


class OOPSHorizonTests(unittest.TestCase):
    def test_update_budget_scales_with_collected_transitions(self):
        agent = OOPS.__new__(OOPS)
        agent.ddpg_config = SimpleNamespace(mini_epochs=None)
        agent.horizon = 1000
        for actors in (1, 2, 64):
            agent.num_actors = actors
            self.assertEqual(agent._updates_per_rollout(), actors * 1000)
        agent.ddpg_config.mini_epochs = 17
        self.assertEqual(agent._updates_per_rollout(), 17)

    def test_reported_rewards_mask_terminal_steps_without_latching(self):
        rewards = torch.tensor([-200., -3., 5., 7.])
        dones = torch.tensor([0, 0, 0, 1])
        info = {'termination': torch.tensor([True, False, False, False])}
        actual = OOPS._reported_rewards(rewards, dones, info)
        torch.testing.assert_close(actual, torch.tensor([0., -3., 5., 0.]))
        torch.testing.assert_close(rewards, torch.tensor([-200., -3., 5., 7.]))
        # Like the reference, a later recovered state can earn reward again.
        actual = OOPS._reported_rewards(rewards, torch.zeros(4), {})
        torch.testing.assert_close(actual, rewards)

    def test_critic_terminal_mask_uses_time_instead_of_environment_done(self):
        agent = OOPS.__new__(OOPS)
        agent.ddpg_config = SimpleNamespace(gamma=0.9, nstep=1)
        agent.critic_target = Mock()
        agent.critic_target.get_q_values.return_value = [torch.full((2, 1), 10.0)]
        agent.critic = Mock()
        agent.critic.get_q_values.return_value = [torch.zeros(2, 1)]
        agent._target_policy_actions = Mock(return_value=torch.zeros(2, 1))
        agent._augmented_state = Mock(return_value=torch.zeros(2, 2))
        agent._pal = Mock(side_effect=lambda td: td.square().mean())
        agent._optimizer_update = Mock(return_value=0.0)
        agent.critic_optim = None
        agent.critic_clip = None
        agent.pal_min_priority = 1.0
        agent.pal_alpha = 0.4
        # Opposite environment flags: timeout was cleared for the terminal
        # sample; an augmented nonterminal time must still bootstrap.
        agent.update_critic(
            {}, torch.zeros(2, 1), torch.ones(2, 1), {},
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[0.001], [0.5]]),
            torch.tensor([[0.0], [0.499]]), None, None,
        )
        torch.testing.assert_close(agent._pal.call_args.args[0], torch.tensor([[-1.0], [-10.0]]))

    def test_rollout_rejects_auto_reset_before_horizon(self):
        agent = OOPS.__new__(OOPS)
        agent.horizon = 2
        agent.num_actors = agent.obs_dim = agent.action_dim = 1
        agent.device = 'cpu'
        agent.state_action = False
        agent.oops_obs_space = {'obs': (1,), 't_to_horizon': (1,)}
        agent.env_autoresets = True
        agent.normalize_input = False
        agent.obs = {'obs': torch.zeros(1, 1)}
        agent.oops_rewarder = Mock()
        env = Mock()
        env.step.return_value = (torch.zeros(1, 1), torch.zeros(1), torch.ones(1), {})
        with self.assertRaisesRegex(RuntimeError, 'reset before the fixed horizon'):
            agent.explore_env(env, 2, random=True)
        agent.oops_rewarder.compute_episode_reward.assert_not_called()

    def test_training_inputs_are_normalized_once_exactly_like_acting_inputs(self):
        agent = OOPS.__new__(OOPS)
        agent.ddpg_config = SimpleNamespace(
            mini_epochs=1, update_actor_interval=1, update_targets_interval=10**9, batch_size=4
        )
        agent.mini_epoch = 0
        agent.horizon = 10
        agent.aug_time = False
        agent.normalize_input = True
        agent.device = 'cpu'
        rms = normalizers.RunningMeanStd((3,), with_clamp=True)
        rms.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
        rms.running_var.copy_(torch.tensor([4.0, 4.0, 4.0]))
        agent.obs_rms = {'obs': rms}

        obs = {'obs': torch.randn(4, 3) + 5.0, 't_to_horizon': torch.rand(4, 1)}
        next_obs = {'obs': torch.randn(4, 3) + 5.0, 't_to_horizon': torch.rand(4, 1)}
        memory = Mock()
        memory.sample_batch.return_value = (obs, torch.zeros(4, 2), torch.zeros(4, 1), next_obs, torch.zeros(4, 1))
        agent.update_critic = Mock(return_value=(torch.tensor(0.0), None))
        agent.update_actor = Mock(return_value=(torch.tensor(0.0), None))

        agent.update_net(memory)

        # update_net must hand RAW observations to the critic/actor updates: the network
        # input is built by `_augmented_state`, which normalizes. Normalizing in update_net
        # too meant training saw normalize(normalize(x)) while acting saw normalize(x).
        critic_obs = agent.update_critic.call_args.args[0]['obs']
        critic_next_obs = agent.update_critic.call_args.args[3]['obs']
        actor_obs = agent.update_actor.call_args.args[0]['obs']
        torch.testing.assert_close(critic_obs, obs['obs'])
        torch.testing.assert_close(critic_next_obs, next_obs['obs'])
        torch.testing.assert_close(actor_obs, obs['obs'])
        train_input = agent._augmented_state({'obs': critic_obs}, obs['t_to_horizon'])
        act_input = agent._augmented_state({'obs': obs['obs']}, obs['t_to_horizon'])
        torch.testing.assert_close(train_input, act_input)

if __name__ == '__main__':
    unittest.main()
