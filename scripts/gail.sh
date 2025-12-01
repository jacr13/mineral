git pull

python spawner.py \
    --task_name gail \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment local  \
    --runtime 4h \
    --no-cleanup \
    --no-sweep \
    --sweep_max 10 \
    --set wandb.project=test-otil-slurm \
    --env_bundle dflex \
    --deploy_now