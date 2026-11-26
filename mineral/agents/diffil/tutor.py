"""TUTOR: MENTOR with a teacher that needs no simulator gradient.

The agent is MENTOR itself (mentor.py, unchanged); this subclass only
  - builds the teacher named by mentor.teacher.kind when it is one of the TUTOR teachers (mentor_gf.GF_TEACHERS,
    which include mentor_hit's): MENTOR.train is run with kind 'rejoin', and the teacher it assigns is replaced by the
    TUTOR teacher, built with the same arguments; any MENTOR kind runs MENTOR's own code path;
  - hands every round's learner rollout to teachers that learn from simulated segments (observe_rollout).
Nothing in mentor.py or the MENTOR configuration is modified, so merges from mentor-wip stay clean.
Stage I is MENTOR's (choose invert.jacobian=fd or broyden with refresh_jacobian=fd for no gradient).
"""

from omegaconf import OmegaConf

from .mentor import MENTOR, RejoinTeacher
from .mentor_gf import GF_TEACHERS


class TUTOR(MENTOR):
    def __init__(self, full_cfg, **kwargs):
        self._tutor_kind = None
        self._teacher = None
        super().__init__(full_cfg, **kwargs)

    @property
    def teacher(self):
        return self._teacher

    @teacher.setter
    def teacher(self, value):
        kind = self._tutor_kind
        if kind is not None and type(value) is RejoinTeacher:  # the placeholder MENTOR.train just built
            cfg = self.mentor_config.teacher
            value = GF_TEACHERS[kind](
                self.spec, self.S, self.S1, self.A_hat, self.lengths, int(cfg.batch), self.device,
                self.counters['teacher'], cfg,
            )
            print(f'[TUTOR] teacher: {kind} ({type(value).__name__})')
        self._teacher = value

    def train(self):
        cfg = self.mentor_config.teacher
        kind = str(cfg.get('kind', 'rejoin'))
        if kind in GF_TEACHERS:
            if self.full_state:
                raise ValueError(f'teacher.kind={kind} is not wired for full-state MENTOR')
            self._tutor_kind = kind
            OmegaConf.set_struct(cfg, False)
            cfg.kind = 'rejoin'  # MENTOR builds its rejoin teacher; the setter replaces it
            cfg.tutor_kind = kind
        return super().train()

    def _teacher_iter_stats(self):
        stats = super()._teacher_iter_stats()
        if hasattr(self._teacher, 'round_stats'):  # diagnostics of the teachers that learn a model
            stats.update(self._teacher.round_stats())
        return stats

    def collect(self):
        visited = super().collect()
        if hasattr(self._teacher, 'observe_rollout'):
            self._teacher.observe_rollout(self.last_rollout)
        return visited
