import os

import torch


def get_demos(device, path="", n_trajs=None, subsample_factor=1):
    """Load and optionally subsample demonstration data from a given path.

    Args:
        device (torch.device): Device to load the data onto.
        path (str): Path to the demonstration data file.
        n_trajs (int): Number of demonstration trajectories to load.
        subsample_factor (int): Factor for time subsampling (dim=1).
            1 = no subsampling; >1 keeps every nth step with a random start.

    Returns:
        dict: keys include "obs", optional "next_obs", "act", "done", "rew",
              and "expert_return" (mean return over first n_trajs, no subsample).
    """
    assert os.path.exists(path), f"Demos path: {path} does not exist"
    assert n_trajs is not None, "n_trajs must be specified"
    assert subsample_factor >= 1, "subsample_factor must be >= 1"

    n_trajs = int(n_trajs)
    subsample_factor = int(subsample_factor)

    demos_raw = torch.load(path, map_location=device)

    # Random start offsets per trajectory, reused for all fields
    starts = None
    if subsample_factor > 1:
        starts = torch.randint(0, subsample_factor, (n_trajs,), device=device)

    def _slice(tensor):
        return tensor[:n_trajs, ...]

    def _subsample(tensor):
        # No time dimension → skip
        if tensor.ndim < 2 or subsample_factor == 1:
            return tensor
        # Build per-traj views with the shared random starts
        subs = [tensor[i, int(starts[i].item()) :: subsample_factor, ...] for i in range(tensor.shape[0])]
        # Trim to the shortest length so stacking works
        min_len = min(t.shape[0] for t in subs)
        subs = [t[:min_len, ...] for t in subs]
        return torch.stack(subs, dim=0)

    def slice_and_subsample(t):
        t = _slice(t.to(device))
        return _subsample(t) if subsample_factor > 1 else t

    def process_obs_like(x):
        if isinstance(x, dict):
            return {k: slice_and_subsample(v) for k, v in x.items()}
        else:
            return slice_and_subsample(x)

    demos = {}

    # obs (dict or tensor)
    demos["obs"] = process_obs_like(demos_raw["obs"])

    # next_obs (dict or tensor)
    demos["next_obs"] = process_obs_like(demos_raw["next_obs"])

    # act, done, rew
    for key in ("act", "done", "rew"):
        demos[key] = slice_and_subsample(demos_raw[key])

    # Ensure that done=1 stays 1 after first terminal in subsampled data
    done = demos["done"]
    print(done)
    for i in range(done.shape[0]):
        done_idx = (done[i] == 1).nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            first_done = done_idx[0].item()
            done[i, first_done + 1 :] = 1
    demos["done"] = done

    # Mean expert return over selected trajectories (no subsampling, but first n_trajs)
    rew = _slice(demos_raw["rew"].to(device))
    demos["expert_return"] = rew.sum(dim=1).mean().item()

    return demos
