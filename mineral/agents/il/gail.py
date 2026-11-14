"""GAIL: Generative Adversarial Imitation Learning (Ho & Ermon, 2016), on PPO.

Same as GAIfO with a state-action discriminator D(s, a): needs the expert actions (not state-only); reference only.
"""

from .gaifo import GAIfO


class GAIL(GAIfO):
    USE_ACTIONS = True
