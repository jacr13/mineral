def make_envs(config):
    from isaacgymenvs.tasks import isaacgym_task_map
    from isaacgymenvs.utils.reformat import omegaconf_to_dict

    # `xvfb-run -a -s "-screen 0 256x256x24 -ac +extension GLX +render -noreset" python ...`
    # set virtual_screen_capture=True and headless=False to get IsaacGym GUI window
    # if you get `OSError: Pillow was built without XCB support`, then `pip install -U Pillow`
    # (switch to pip instead of conda package)  # https://stackoverflow.com/a/66682282

    if config.env_render:
        headless, virtual_screen_capture = False, True
    else:
        headless, virtual_screen_capture = config.headless, False

    env = isaacgym_task_map[config.task_name](
        cfg=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        rl_device=config.rl_device,
        graphics_device_id=config.graphics_device_id,
        headless=headless,
        virtual_screen_capture=virtual_screen_capture,
        force_render=False,
    )
    return env


def import_isaacgym():
    # https://github.com/NVlabs/sim-web-visualizer/blob/main/example/isaacgym/train_isaacgym_remote_server.ipynb
    import os
    import sys
    from ctypes import cdll
    from pathlib import Path

    is_conda = 'CONDA_PREFIX' in os.environ or 'CONDA_DEFAULT_ENV' in os.environ
    if is_conda:
        version_info = sys.version_info
        if version_info.major == 3 and version_info.minor >= 8:
            conda_lib_path = (
                Path(sys.executable).parent.parent / f"lib/libpython{version_info.major}.{version_info.minor}.so.1.0"
            )
        else:
            conda_lib_path = (
                Path(sys.executable).parent.parent / f"lib/libpython{version_info.major}.{version_info.minor}m.so.1.0"
            )
        python_lib = cdll.LoadLibrary(str(conda_lib_path))
        print(f"Load Python lib {conda_lib_path}")

    import isaacgym
    import torch
