git pull

python spawner.py \
    --task_name gail \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm  \
    --runtime 4h \
    --no-cleanup \
    --no-sweep \
    --sweep_max 10 \
    --set wandb.project=test-otil-slurm \
    --set agent.gail.expert_batch_size=516 \
    --env_bundle dflex \
    --deploy_now