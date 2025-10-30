git pull

python spawner.py \
    --task_name sapo \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm  \
    --runtime 12h \
    --no-cleanup \
    --sweep \
    --sweep_max 10 \
    --deploy_now 