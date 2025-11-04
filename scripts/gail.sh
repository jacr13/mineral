git pull

python spawner.py \
    --task_name gail \
    --no-docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment local  \
    --runtime 12h \
    --no-cleanup \
    --no-sweep \
    --sweep_max 10 \
    --set wandb.project=test-rewarped-local \
    --deploy_now 