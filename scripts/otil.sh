git pull

python spawner.py \
    --task_name otil \
    --docker \
    --docker_image /home/users/c/candidor/docker/mineral.sif \
    --deployment slurm \
    --runtime 6h \
    --no-cleanup \
    --sweep \
    --sweep_max 150 \
    --set wandb.project=sweep-dflex_snu_humanoid-slurm-new \
    --env_files dflex_snu_humanoid.yaml \
    --deploy_now