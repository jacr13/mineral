from .agent import Agent


class Template(Agent):
    r"""Template Agent."""

    def __init__(self, full_cfg, **kwargs):
        self.network_config = full_cfg.agent.network
        self.template_config = full_cfg.agent.template
        self.num_actors = self.template_config.num_actors
        self.max_agent_steps = int(self.template_config.max_agent_steps)
        super().__init__(full_cfg, **kwargs)

    def train(self):
        pass

    def eval(self):
        pass

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def save(self, f):
        pass

    def load(self, f, ckpt_keys=''):
        pass
