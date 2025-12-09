git pull

python spawner.py \
    --task_name ild \
    --no-docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment local  \
    --runtime 4h \
    --no-cleanup \
    --no-sweep \
    --sweep_max 10 \
    --set wandb.project=test-rewarped-local \
    --env_bundle dflex \
    --no-deploy_now