"""Regression checks for periodic checkpoint retention."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mineral.agents.agent import Agent


class AgentCheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.ckpt_dir = Path(directory.name)
        self.agent = Agent.__new__(Agent)
        self.agent.ckpt_dir = directory.name
        self.agent.ckpt_every = 1
        self.agent.best_stat = None
        self.agent.agent_steps = 0
        self.agent.save = lambda filename: Path(filename).write_text('checkpoint')

    def test_periodic_checkpoints_keep_last_ten_and_preserve_best(self):
        for epoch in range(25):
            self.agent.epoch = epoch
            self.agent._checkpoint_save(25 - epoch)
            self.assertEqual(len(list(self.ckpt_dir.glob('epochs*.pth'))), min(epoch + 1, 10))

        expected = {f'epochs{epoch}_steps0k_rewards{26 - epoch:.2f}.pth' for epoch in range(16, 26)}
        self.assertEqual({path.name for path in self.ckpt_dir.glob('epochs*.pth')}, expected)
        self.assertTrue((self.ckpt_dir / 'best_rewards25.00.pth').is_file())
        self.assertEqual(os.readlink(self.ckpt_dir / 'latest.pth'), 'epochs25_steps0k_rewards1.00.pth')
        self.assertEqual((self.ckpt_dir / 'latest.pth').read_text(), 'checkpoint')

    def test_resumed_run_prunes_existing_checkpoints_and_repairs_latest(self):
        for epoch in range(1, 21):
            (self.ckpt_dir / f'epochs_{epoch}_steps_0k_rewards_1.00.pth').touch()
        (self.ckpt_dir / 'latest.pth').symlink_to('missing.pth')
        (self.ckpt_dir / 'manual.pth').touch()
        (self.ckpt_dir / 'epochs_notes.pth').touch()
        self.agent.epoch = 20
        self.agent._checkpoint_save(1, sep='_')

        expected = {f'epochs_{epoch}_steps_0k_rewards_1.00.pth' for epoch in range(12, 22)}
        self.assertEqual({path.name for path in self.ckpt_dir.glob('epochs_*.pth')} - {'epochs_notes.pth'}, expected)
        self.assertTrue((self.ckpt_dir / 'manual.pth').exists())
        self.assertTrue((self.ckpt_dir / 'epochs_notes.pth').exists())
        self.assertTrue((self.ckpt_dir / 'latest.pth').is_file())

    def test_failed_save_does_not_prune_checkpoints_or_change_latest(self):
        for epoch in range(10):
            self.agent.epoch = epoch
            self.agent._checkpoint_save(1)
        before = set(self.ckpt_dir.iterdir())
        latest = os.readlink(self.ckpt_dir / 'latest.pth')
        self.agent.epoch = 10
        self.agent.save = Mock(side_effect=OSError('save failed'))
        with self.assertRaisesRegex(OSError, 'save failed'):
            self.agent._checkpoint_save(1)
        self.assertEqual(set(self.ckpt_dir.iterdir()), before)
        self.assertEqual(os.readlink(self.ckpt_dir / 'latest.pth'), latest)


if __name__ == '__main__':
    unittest.main()
