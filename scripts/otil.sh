git pull

python spawner.py \
    --task_name otil \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm  \
    --runtime 12h \
    --no-cleanup \
    --sweep \
    --sweep_max 10 \
    --set wandb.project=test-otil-slurm \
    --env_bundle dflex \
    --no-deploy_now