git pull

python spawner.py \
    --task_name sapo/dflex_ant.yaml \
    --docker \
    --docker_image mineral.sif \
    --deployment slurm  \
    --runtime 10m \
    # --deploy_now \
    --no-cleanup \
    --set agent.shac.max_epoch=2