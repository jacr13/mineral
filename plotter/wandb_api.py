import wandb


def get_group_runs(run_url: str):
    run_url_parts = run_url.split("/")
    api = wandb.Api()
    entity = run_url_parts[3]
    project = run_url_parts[4]
    group_name = run_url_parts[6]
    print(f"Fetching runs for group: {group_name} in project: {project} of entity: {entity}")
    runs = api.runs(f"{entity}/{project}", filters={"group": group_name})

    results = []
    for run in runs:
        config = dict(run.config)
        results.append(
            {
                "run_id": run.id,
                "name": run.name,
                "config": config,
                "logdir": config["logdir"],
            }
        )

    return results
