import os

import torch

folder = "dflex"

for path in os.listdir(folder):
    path = os.path.join(folder, path)
    data = torch.load(path, map_location=torch.device("cpu"), weights_only=True)

    episode_obs = {k: [] for k in data[0]["obs"].keys()}
    episode_act = []
    episode_next_obs = {k: [] for k in data[0]["next_obs"].keys()}
    episode_rew = []
    episode_done = []

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
        if data[i]["act"].shape[0] == max_t:
            for k, v in data[i]["obs"].items():
                episode_obs[k].append(v)
                episode_next_obs[k].append(data[i]["next_obs"][k])
            episode_act.append(data[i]["act"])
            episode_rew.append(data[i]["rew"])
            episode_done.append(data[i]["done"])
        else:
            print("Skipping", i)

    ep_obs = {k: torch.stack(v) for k, v in episode_obs.items()}
    ep_next_obs = {k: torch.stack(v) for k, v in episode_next_obs.items()}
    ep_act = torch.stack(episode_act)
    ep_rew = torch.stack(episode_rew)
    ep_done = torch.stack(episode_done)

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

    cleaned_path = (
        "_".join(path.split("/")[-1].split("_")[:2])
        + f"_demos{ep_rew.shape[0]}_return{int(ep_rew.sum(-1).mean())}_len{ep_rew.shape[1]}.pt"
    )
    print(cleaned_path)

    torch.save(
        {
            "obs": ep_obs,
            "act": ep_act,
            "next_obs": ep_next_obs,
            "rew": ep_rew,
            "done": ep_done,
            # "joint_q": ep_joint_q,
            # "joint_qd": ep_joint_qd,
        },
        cleaned_path,
    )
