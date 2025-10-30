git pull

python spawner.py \
    --task_name sapo/dflex_ant.yaml \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm  \
    --runtime 10m \
    --no-cleanup \
    --set agent.shac.max_epochs=2 \
    --deploy_now 
