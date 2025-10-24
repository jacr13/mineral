import torch

path = "dflex/DFlex_ant_demos128_epochs4882_steps10000384_return14008_len1500.pt"
data = torch.load(path, map_location=torch.device("cpu"), weights_only=True)

episode_obs = []
episode_act = []
episode_next_obs = []
episode_rew = []
episode_done = []
episode_joint_q = []
episode_joint_qd = []


for i in data:
    print(i, data[i]["obs"].shape[0])
    if data[i]["obs"].shape[0] == 1000:
        episode_obs.append(data[i]["obs"])
        episode_act.append(data[i]["act"])
        episode_next_obs.append(data[i]["next_obs"])
        episode_rew.append(data[i]["rew"])
        episode_done.append(data[i]["done"])
        # episode_joint_q.append(data[i]["joint_q"])
        # episode_joint_qd.append(data[i]["joint_qd"])


ep_obs = torch.stack(episode_obs)
ep_act = torch.stack(episode_act)
ep_next_obs = torch.stack(episode_next_obs)
ep_rew = torch.stack(episode_rew)
ep_done = torch.stack(episode_done)
# ep_joint_q = torch.stack(episode_joint_q)
# ep_joint_qd = torch.stack(episode_joint_qd)


print(
    ep_obs.shape,
    ep_act.shape,
    ep_next_obs.shape,
    ep_rew.shape,
    ep_done.shape,
    # ep_joint_q.shape,
    # ep_joint_qd.shape,
)


print("ep_obs", ep_obs.shape, ep_obs.sum(), ep_obs.mean(), ep_obs.std())
print("ep_act", ep_act.shape, ep_act.sum(), ep_act.mean(), ep_act.std())
print(
    "ep_next_obs",
    ep_next_obs.shape,
    ep_next_obs.sum(),
    ep_next_obs.mean(),
    ep_next_obs.std(),
)
print("ep_rew", ep_rew.shape, ep_rew.sum(), ep_rew.mean(), ep_rew.std())
print("ep_done", ep_done.shape)
# print("ep_joint_q", ep_joint_q.shape)
# print("ep_joint_qd", ep_joint_qd.shape)

print(ep_rew.sum(-1).mean())

cleaned_path = (
    "_".join(path.split("/")[-1].split("_")[:2])
    + f"_demos{ep_obs.shape[0]}_return{int(ep_rew.sum(-1).mean())}_len{ep_obs.shape[1]}.pt"
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
