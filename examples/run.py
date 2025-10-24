import os

import hydra

if __name__ == '__main__':
    c = []
    hydra.main(
        config_name='config',
        config_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../mineral/cfgs'),
        version_base='1.1',
    )(lambda x: c.append(x))()
    config = c[0]

    from mineral.scripts import run

    run.main(config)
