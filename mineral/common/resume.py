"""Snapshot and restore of an agent's whole training state (resumable checkpoints, ``ckpt/resume.pth``).

A resume checkpoint lets a run continue exactly where it stopped: training it to N steps, saving, and resuming it to
N + M steps gives the same result as one uninterrupted run to N + M steps (``scripts/resume_check.py``). It holds

- every ``nn.Module`` and optimizer of the agent (weights, target networks, normalisers, Adam moments, learning rates),
- every tensor attribute (``log_alpha``, rollout buffers, episode trackers, ...), replay buffers included,
- the agent's plain Python values that changed since the agent was built (counters, schedules, histories),
- the environment's tensors and its simulation state (DFlex ``state``: joint positions and velocities, ...),
- the random number generators (torch CPU, every CUDA device, numpy, python, and the agent's own torch Generators).

It is taken generically from the objects' attributes, so new agents are covered without code. Plain values that were
unchanged since construction are taken from the new run's configuration instead: they are settings (budgets,
intervals), and a resumed run may change them, e.g. a larger ``max_agent_steps`` to train longer. Objects of other
packages (configs, writers, the job clock, the evaluation env, closures) are not saved.
"""
import collections
import os
import pickle
import random

import numpy as np
import torch

PLAIN = (bool, int, float, complex, str, bytes, type(None), np.generic)
# agent attributes never saved: the run's environment and services, and the settings a resumed run may change
AGENT_SKIP = {
    'env', 'full_cfg', 'writer', 'tb_summary_writer', 'job_clock', 'datasets', 'accelerator', 'timer',
    'logdir', 'ckpt_dir', 'tb_dir', 'device', 'rank', 'rank_size', 'multi_gpu',
    '_protocol_env', '_protocol_t0', '_protocol_stopped_by_clock', '_protocol_stopped_by_train_time',
    'max_agent_steps', 'max_epochs', 'protocol_max_train_seconds', 'eval_every_steps', 'protocol_eval',
    'ckpt_every', 'print_every', 'eval_every',
}
# environment attributes never saved (the simulation model and its builders are rebuilt from the config)
ENV_SKIP = {'model', 'builder', 'integrator', 'renderer', 'stage', 'obs_space', 'act_space', 'device'}


class _Skip:
    pass


SKIP = _Skip()


def _is_config(v):
    try:
        from omegaconf import DictConfig, ListConfig

        return isinstance(v, (DictConfig, ListConfig))
    except ImportError:  # pragma: no cover
        return False


def _ours(v):
    return type(v).__module__.split('.')[0] == 'mineral' and hasattr(v, '__dict__')


def _plain(v):
    """True for values stored as they are (numbers, strings, numpy arrays, containers of those)."""
    if isinstance(v, PLAIN) or isinstance(v, np.ndarray):
        return True
    if isinstance(v, (list, tuple, collections.deque)):
        return all(_plain(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, PLAIN) and _plain(x) for k, x in v.items())
    return False


def _equal(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and bool(
                np.array_equal(a, b))
        return type(a) is type(b) and pickle.dumps(a) == pickle.dumps(b)
    except Exception:
        return False


def plain_marks(obj, depth=0, skip=AGENT_SKIP, seen=None):
    """{path: pickled value} of the plain values reachable from ``obj`` (the values at construction)."""
    seen = set() if seen is None else seen
    out = {}
    if id(obj) in seen or depth > 4:
        return out
    seen.add(id(obj))
    for k, v in vars(obj).items():
        if depth == 0 and k in skip:
            continue
        if _plain(v):
            try:
                out[k] = pickle.dumps(v)
            except Exception:
                pass
        elif _ours(v) and not isinstance(v, (torch.nn.Module, torch.optim.Optimizer)):
            out.update({f'{k}.{kk}': vv for kk, vv in plain_marks(v, depth + 1, skip, seen).items()})
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if _ours(vv):
                    out.update({f'{k}[{kk!r}].{a}': b for a, b in plain_marks(vv, depth + 1, skip, seen).items()})
    return out


def _encode(v, path, marks, depth, seen):
    if isinstance(v, torch.nn.Module):
        return ('module', v.state_dict())
    if isinstance(v, torch.optim.Optimizer):
        return ('optim', v.state_dict())
    if torch.is_tensor(v):
        return ('tensor', v.detach().clone())
    if isinstance(v, torch.Generator):
        return ('generator', v.get_state())
    if _is_config(v) or callable(v) and not _ours(v):
        return SKIP
    if _plain(v):
        mark = marks.get(path)
        if mark is not None:
            try:
                if pickle.dumps(v) == mark:
                    return SKIP  # unchanged since construction: a setting, taken from the new run's config
            except Exception:
                pass
        if isinstance(v, collections.deque):
            return ('deque', (list(v), v.maxlen))
        return ('value', v)
    if depth > 4 or id(v) in seen:
        return SKIP
    if isinstance(v, (list, tuple)):
        enc = [_encode(x, f'{path}[{i}]', marks, depth + 1, seen) for i, x in enumerate(v)]
        if any(e is SKIP for e in enc):
            return SKIP
        return ('list' if isinstance(v, list) else 'tuple', enc)
    if isinstance(v, dict):
        enc = {}
        for k, x in v.items():
            if not isinstance(k, PLAIN):
                return SKIP
            e = _encode(x, f'{path}[{k!r}]', marks, depth + 1, seen)
            if e is not SKIP:
                enc[k] = e
        return ('dict', enc)
    if _ours(v):
        seen.add(id(v))
        hook = getattr(v, 'resume_state', None)
        if callable(hook):
            return ('hook', hook())
        enc = {}
        for k, x in vars(v).items():
            e = _encode(x, f'{path}.{k}', marks, depth + 1, seen)
            if e is not SKIP:
                enc[k] = e
        return ('obj', enc)
    return SKIP


def _decode(cur, enc, path, missing):
    kind, val = enc
    if kind == 'module':
        if isinstance(cur, torch.nn.Module):
            cur.load_state_dict(val)
            return cur
        missing.append(path)
        return cur
    if kind == 'optim':
        if isinstance(cur, torch.optim.Optimizer):
            cur.load_state_dict(val)
            return cur
        missing.append(path)
        return cur
    if kind == 'tensor':
        if torch.is_tensor(cur) and cur.shape == val.shape and cur.dtype == val.dtype:
            with torch.no_grad():
                cur.copy_(val.to(cur.device))
            return cur
        return val
    if kind == 'generator':
        if isinstance(cur, torch.Generator):
            cur.set_state(val)
            return cur
        missing.append(path)
        return cur
    if kind == 'value':
        return val
    if kind == 'deque':
        return collections.deque(val[0], maxlen=val[1])
    if kind in ('list', 'tuple'):
        out = []
        for i, e in enumerate(val):
            c = cur[i] if isinstance(cur, (list, tuple)) and i < len(cur) else None
            out.append(_decode(c, e, f'{path}[{i}]', missing))
        if isinstance(cur, list) and len(cur) == len(out):
            cur[:] = out
            return cur
        return out if kind == 'list' else tuple(out)
    if kind == 'dict':
        if not isinstance(cur, dict):
            cur = {}
        for k, e in val.items():
            c = cur[k] if (k in cur or isinstance(cur, collections.defaultdict)) else None
            cur[k] = _decode(c, e, f'{path}[{k!r}]', missing)
        return cur
    if kind == 'hook':
        if cur is None or not callable(getattr(cur, 'resume_load', None)):
            missing.append(path)
            return cur
        cur.resume_load(val)
        return cur
    if kind == 'obj':
        if cur is None:
            missing.append(path)
            return cur
        for k, e in val.items():
            setattr(cur, k, _decode(getattr(cur, k, None), e, f'{path}.{k}', missing))
        return cur
    raise ValueError(f'resume: unknown entry {kind} at {path}')


def agent_state(agent, marks):
    seen = set()
    out = {}
    skipped = []
    for k, v in vars(agent).items():
        if k in AGENT_SKIP or k.startswith('_resume'):
            continue
        e = _encode(v, k, marks, 0, seen)
        if e is not SKIP:
            out[k] = e
        else:
            skipped.append(f'{k}:{type(v).__name__}')
    if os.environ.get('MINERAL_RESUME_DEBUG'):  # what is not saved (settings unchanged since construction, services)
        print('[resume] not saved:', ' '.join(skipped))
    return out


def load_agent_state(agent, state):
    missing = []
    for k, e in state.items():
        setattr(agent, k, _decode(getattr(agent, k, None), e, k, missing))
    return missing


def env_state(env, marks):
    """Tensors and changed plain values of the environment, and the tensors of its simulation state."""
    out = {}
    for k, v in vars(env).items():
        if k in ENV_SKIP:
            continue
        if k == 'state' and hasattr(v, '__dict__'):
            out['state'] = ('sim', {kk: vv.detach().clone() for kk, vv in vars(v).items() if torch.is_tensor(vv)})
            continue
        if torch.is_tensor(v):
            out[k] = ('tensor', v.detach().clone())
        elif isinstance(v, dict) and all(torch.is_tensor(x) or _plain(x) for x in v.values()):
            out[k] = ('dict', {kk: ('tensor', x.detach().clone()) if torch.is_tensor(x) else ('value', x)
                               for kk, x in v.items()})
        elif _plain(v) and (k not in marks or pickle.dumps(v) != marks[k]):
            out[k] = ('value', v)
    return out


def _fresh(saved, cur):
    """A copy of the saved tensor with the requires_grad flag of the env's current one (DFlex sets it on states)."""
    t = saved.clone().to(cur.device if torch.is_tensor(cur) else saved.device)
    if torch.is_tensor(cur) and cur.requires_grad and t.is_floating_point():
        t.requires_grad_(True)
    return t


def load_env_state(env, state):
    for k, (kind, val) in state.items():
        cur = getattr(env, k, None)
        if kind == 'sim':
            for kk, t in val.items():
                setattr(cur, kk, _fresh(t, getattr(cur, kk, None)))
        elif kind == 'tensor':
            setattr(env, k, _fresh(val, cur))
        elif kind == 'dict':
            setattr(env, k, {kk: _fresh(x, None) if kd == 'tensor' else x for kk, (kd, x) in val.items()})
        else:
            setattr(env, k, val)


def rng_state():
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def load_rng_state(s):
    torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])
    np.random.set_state(s['numpy'])
    random.setstate(s['python'])
