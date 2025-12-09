git pull

python spawner.py \
    --task_name otil \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm \
    --runtime 4h \
    --no-cleanup \
    --sweep \
    --sweep_max 50 \
    --set wandb.project=sweep-dflex_ant-slurm \
    --env_files dflex_ant.yaml \
    --deploy_now