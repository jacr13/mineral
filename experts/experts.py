import os

import torch

folder = "dflex"

for path in os.listdir(folder):
    full_path = os.path.join(folder, path)
    data = torch.load(full_path, map_location=torch.device("cpu"), weights_only=True)

    episode_obs = {k: [] for k in data[0]["obs"].keys()}
    episode_act = []
    episode_next_obs = {k: [] for k in data[0]["next_obs"].keys()}
    episode_rew = []
    episode_done = []
    episode_lens = []

    # check max shape
    shapes = {}
    for _k, v in data.items():
        if isinstance(v, dict):
            for _k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    if v2.shape[0] not in shapes:
                        shapes[v2.shape[0]] = 1
                    else:
                        shapes[v2.shape[0]] += 1
                elif isinstance(v2, dict):
                    for _k3, v3 in v2.items():
                        if isinstance(v3, torch.Tensor):
                            if v3.shape[0] not in shapes:
                                shapes[v3.shape[0]] = 1
                            else:
                                shapes[v3.shape[0]] += 1
                        else:
                            raise ValueError
                else:
                    raise ValueError
        else:
            raise ValueError

    max_t = max(shapes.keys())
    print(max_t)

    for i in data:
        print(i, data[i]["act"].shape[0])
        t = data[i]["act"].shape[0]

        # record original length
        episode_lens.append(t)

        # pad obs and next_obs to max_t along time dimension
        for k, v in data[i]["obs"].items():
            padded = torch.zeros((max_t, *v.shape[1:]), dtype=v.dtype)
            padded[:t] = v
            episode_obs[k].append(padded)

            v_next = data[i]["next_obs"][k]
            padded_next = torch.zeros((max_t, *v_next.shape[1:]), dtype=v_next.dtype)
            padded_next[:t] = v_next
            episode_next_obs[k].append(padded_next)

        # pad act, rew, done
        act = data[i]["act"]
        padded_act = torch.zeros((max_t, *act.shape[1:]), dtype=act.dtype)
        padded_act[:t] = act
        episode_act.append(padded_act)

        rew = data[i]["rew"]
        padded_rew = torch.zeros((max_t, *rew.shape[1:]), dtype=rew.dtype)
        padded_rew[:t] = rew
        episode_rew.append(padded_rew)

        done = data[i]["done"]
        padded_done = torch.zeros((max_t, *done.shape[1:]), dtype=done.dtype)
        padded_done[:t] = done
        episode_done.append(padded_done)

    ep_obs = {k: torch.stack(v) for k, v in episode_obs.items()}
    ep_next_obs = {k: torch.stack(v) for k, v in episode_next_obs.items()}
    ep_act = torch.stack(episode_act)
    ep_rew = torch.stack(episode_rew)
    ep_done = torch.stack(episode_done)
    ep_len = torch.tensor(episode_lens, dtype=torch.long)

    for k, v in ep_obs.items():
        print(k, v.shape)
    for k, v in ep_next_obs.items():
        print(k, v.shape)
    print(
        ep_act.shape,
        ep_rew.shape,
        ep_done.shape,
        # ep_joint_q.shape,
        # ep_joint_qd.shape,
    )

    print(ep_rew.sum(-1).mean())

    avg_len = int(ep_len.float().mean().item())
    cleaned_path = (
        "_".join(path.split("/")[-1].split("_")[:2])
        + f"_demos{ep_rew.shape[0]}_return{int(ep_rew.sum(-1).mean())}_len{avg_len}.pt"
    )
    print(cleaned_path)

    torch.save(
        {
            "obs": ep_obs,
            "act": ep_act,
            "next_obs": ep_next_obs,
            "rew": ep_rew,
            "done": ep_done,
            "lengths": ep_len,
            # "joint_q": ep_joint_q,
            # "joint_qd": ep_joint_qd,
        },
        cleaned_path,
    )
