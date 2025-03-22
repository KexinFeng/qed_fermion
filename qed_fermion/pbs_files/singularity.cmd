#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --time=0-00:30:00
#SBATCH --qos=debug
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fengx463@hku.hk     #    Email address to receive notification
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=./report/%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
#SBATCH --error=./report/%x_%j.err             #    Standard error log as $job_name_$job_id.errs

cd ${SLURM_SUBMIT_DIR}
module load singularity
module load cuda/12.3
scontrol show job $SLURM_JOB_ID  # print some info

singularity instance start --nv \
    --no-home -f \
    --overlay overlay.img \
    --bind ~/.vscode-server:$HOME/.vscode-server \
    --bind ~/mount_folder:/opt/mount_folder \
    cuda_12.3.0-devel-rockylinux8.sif zen_dirac

sleep infinity