"""Silence known, harmless third-party notices so run logs show only mineral's output.

Each filter matches one specific message; anything else still prints.
"""

import sys
import types
import warnings

# Exact third-party messages that need no action here.
_IGNORED = (
    # dflex builds gym spaces from float64 bounds; gym casts them to float32.
    (r'.*Box bound precision lowered by casting to float32', UserWarning, ''),
    # dflex/model.py converts a list of numpy arrays to a tensor once at model setup.
    (r'Creating a tensor from a list of numpy\.ndarrays is extremely slow', UserWarning, r'dflex'),
    # config.yaml searches IsaacGymEnvs' configs, which are optional (not installed with uv).
    (r'provider=hydra\.searchpath in main, path=isaacgymenvs\.cfg is not available', UserWarning, ''),
)


def quiet_third_party():
    """Install the filters and mute gym's import notice; call before importing gym or Hydra."""
    for message, category, module in _IGNORED:
        warnings.filterwarnings('ignore', message=message, category=category, module=module)
    # gym 0.23 (used by dflex) prints a "Gym has been unmaintained" notice from the gym-notices
    # package when imported. An empty notice table keeps that import silent.
    if 'gym' not in sys.modules and 'gym_notices' not in sys.modules:
        notices = types.ModuleType('gym_notices.notices')
        notices.notices = {}
        package = types.ModuleType('gym_notices')
        package.notices = notices
        sys.modules['gym_notices'] = package
        sys.modules['gym_notices.notices'] = notices
