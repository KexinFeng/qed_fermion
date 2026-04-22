#!/bin/bash
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
module load python3/3.9.3_anaconda2021.11_mamba
module load matlab/R2023b
module load gcc/9.2.0
module load cuda/12.1.1
## export PATH=/share1/cuda/12.4/bin/:$PATH
## export LD_LIBRARY_PATH=/share1/cuda/12.4/lib64:$LD_LIBRARY_PATH
## ## export LD_LIBRARY_PATH=/home/fengx463/Downloads:$LD_LIBRARY_PATH
## export CUDA_HOME=/share1/cuda/12.4

nvcc --version
nvidia-smi

pip list | grep torch

/home/fengx463/mount_folder/cuda_pcg/cr.sh

