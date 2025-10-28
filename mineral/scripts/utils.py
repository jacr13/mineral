import hashlib
import json
import os
import random
import string
import subprocess
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .humanhash import humanize

try:
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
except:
    pass  # ignore if already registered


def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def set_np_formatting():
    """Formats numpy print."""
    np.set_printoptions(
        edgeitems=30,
        infstr='inf',
        linewidth=4000,
        nanstr='nan',
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def set_seed(seed, torch_deterministic=False, rank=0):
    import random

    import numpy as np
    import torch

    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        # BUG: https://discuss.pytorch.org/t/deterministic-algorithms-yield-an-error/181809
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def limit_threads(n: int = 1):
    # blosc.set_nthreads(n)
    # if n == 1:
    #     blosc.use_threads = False
    # torch.set_num_threads(n)
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n)


def make_batch_env(num_envs, make_env_fn, parallel='process', device='numpy'):
    from functools import partial as bind

    from ..common.batch_env import BatchEnv
    from ..common.parallel import Parallel

    ctors = []
    for _index in range(num_envs):
        ctor = lambda: make_env_fn()
        if parallel != 'none':
            ctor = bind(Parallel, ctor, parallel)
        ctors.append(ctor)

    envs = [ctor() for ctor in ctors]
    env = BatchEnv(envs, parallel=parallel != 'none', device=device)
    return env


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def generate_salt(length):
    """Generate a random salt with the specified length."""
    characters = string.ascii_letters + string.digits
    salt = "".join(random.choice(characters) for _ in range(length))
    return salt


def get_gitsha():
    _gitsha = "gitSHA_{}"
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        gitsha = _gitsha.format(out.strip().decode("ascii"))
    except OSError:
        gitsha = "noGitSHA"
    return gitsha


def replace_batch_norm_to_global(args):
    if args["batch_norm_all"]:
        batch_norm_keys = [key for key in args.keys() if "_batch_norm" in key]
        for key in batch_norm_keys:
            args[key] = True
    return args


def create_uuid(*args, **kwargs):
    """Builds the uuid of the experiment."""
    uuid = uuid_basis(*args, **kwargs)
    config = kwargs.get("config", None)
    assert config is not None, "config must be specified"

    algo = kwargs.get("algo", None) or config.get("agent", {}).get("algo", None)
    seed = kwargs.get("seed", None) or config.get("seed", None)
    task = kwargs.get("task", None) or config.get("task", {}).get("env", {}).get("env_name", None)

    assert seed is not None, "seed must be specified"
    assert algo is not None, "algo must be specified"
    assert task is not None, "task must be specified"

    # Enrich the uuid with extra information
    uuid = f"{uuid}.{algo}.{get_gitsha()}.{task}"
    uuid += f".seed{str(seed).zfill(2)}"
    return uuid


def uuid_basis(*args, **kwargs):
    method = kwargs.get("method", "human_hash")
    add_salt = kwargs.get("add_salt", False)

    uuid = None
    if method == "syllables":
        uuid = uuid_syllables(*args, **kwargs)
    elif method == "hash":
        uuid = uuid_hash(*args, **kwargs)
    elif method == "human_hash":
        uuid = humanize(uuid_hash(*args, **kwargs), words=5)
    else:
        raise NotImplementedError

    if add_salt:
        salt_len = kwargs.get("salt_len", 4)
        salt = generate_salt(salt_len)
        uuid = f"{uuid}-{salt}"
    return uuid


def get_min_config(*args, **kwargs):
    config = kwargs.get("config", None)
    config_default = kwargs.get("config_default", None)
    use_min_config = kwargs.get("use_min_config", False)
    assert config is not None, "config must be specified"
    if not use_min_config:
        return dict(sorted(config.items()))

    assert config_default is not None, "config_default must be specified if use_min_config is True"

    config_min = {key: value for key, value in config.items() if key in config_default and value != config_default[key]}
    return dict(sorted(config_min.items()))


def uuid_hash(*args, **kwargs):
    min_config = get_min_config(*args, **kwargs)
    uuid = dict_hash(min_config)
    return uuid


def uuid_syllables(*args, **kwargs):
    """Randomly create a semi-pronounceable uuid."""
    num_syllables = kwargs.get("num_syllables", 2)
    num_parts = kwargs.get("num_parts", 3)
    part1 = [
        "s",
        "t",
        "r",
        "ch",
        "b",
        "c",
        "w",
        "z",
        "h",
        "k",
        "p",
        "ph",
        "sh",
        "f",
        "fr",
    ]
    part2 = ["a", "oo", "ee", "e", "u", "er"]
    seps = ["_"]  # [ '-', '_', '.']
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for i in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for i in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2, strict=False):
            result += part1[i1] + part2[i2]
    return result
