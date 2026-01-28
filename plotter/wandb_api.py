import wandb
    
API = wandb.Api()

def get_group_runs(run_url: str):
    run_url_parts = run_url.split("/")

    entity = run_url_parts[3]
    project = run_url_parts[4]
    group_name = run_url_parts[6]
    print(f"Fetching runs for group: {group_name} in project: {project} of entity: {entity}")
    runs = API.runs(f"{entity}/{project}", filters={"group": group_name})

    results = []
    for run in runs:
        config = dict(run.config)
        results.append(
            {
                "run_id": run.id,
                "name": run.name,
                "project": run.project,
                "entity": run.entity,
                "config": config,
                "logdir": config["logdir"],
            }
        )

    return results

def get_rew_steps_times(
        entity, 
        project,
        run_id,
        samples=100000,
        keys=["train_scores/episode_rewards", "_runtime"], x_axis="_step",
):
    run = API.run(f"{entity}/{project}/{run_id}")

    pd = run.history(
        keys=keys,
        samples=samples,
        x_axis=x_axis
    )

    ep_rew = pd["train_scores/episode_rewards"].to_numpy()
    steps = pd["_step"].to_numpy()
    times = pd["_runtime"].to_numpy()
    return ep_rew, steps, times

if __name__ == "__main__":
    entity = "jacr"
    project = "OTIL_SAPO-dflex_hopper-slurm-new"
    run_id = "harry-nevada-yellow-freddie-magnesium-CBqP.OTIL.DFlex_hopper_64.seed1000"

    get_rew_steps_times(entity, project, run_id)

