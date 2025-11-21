#!/bin/bash -l
#SBATCH -J qwen
#SBATCH --array=1-2
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=chenxiang.zhang@uni.lu
#SBATCH --ntasks-per-node=1
#SBATCH --account=p200535
#SBATCH --qos=default
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=0-08:00:00
#SBATCH -o slurm-%x-%j_%a.out

echo -e "--------------------------------"
echo -e "Start:\t $(date)"
echo -e "JobID:\t ${SLURM_JOBID}"
echo -e "TaskID:\t ${SLURM_ARRAY_TASK_ID}"
echo -e "Node:\t ${SLURM_NODELIST}"
echo -e "--------------------------------\n"

# Your more useful application can be started below!
micromamba activate lm


id=$((SLURM_ARRAY_TASK_ID - 1))
sweep=(1e-3 1e-6)
lr=${sweep[$id]}

for s in 42 43; do
    python3 train.py \
        --lr $lr \
        --seed $s \
        --output_dir qwen2/lr${lr}_${s}
done