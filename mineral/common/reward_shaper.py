class RewardShaper:
    def __init__(
        self,
        fn='scale',
        scale=1.0,
    ):
        self.fn = fn
        self.scale = scale

    def __call__(self, rewards):
        if self.fn == 'scale':
            rewards = rewards * self.scale
        else:
            raise NotImplementedError(self.fn)
        return rewards
