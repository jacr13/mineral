git pull

python spawner.py \
    --task_name otil \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm \
    --runtime 4h \
    --no-cleanup \
    --sweep \
    --sweep_max 100 \
    --set wandb.project=sweep-dflex_humanoid-slurm \
    --env_files dflex_humanoid.yaml \
    --deploy_now