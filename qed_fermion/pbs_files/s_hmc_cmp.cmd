#!/bin/bash
#SBATCH --mail-type=END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fengx463@hku.hk     #    Email address to receive notification
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=./report_cmp_large9_Gflux/%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
#SBATCH --error=./report_cmp_large9_Gflux/%x_%j.err             #    Standard error log as $job_name_$job_id.errs

cd ${SLURM_SUBMIT_DIR}
module load python/3.9.7
module load matlab/r2023b
export LD_LIBRARY_PATH=/home/fengx463/Downloads:$LD_LIBRARY_PATH

python3 ../hmc_sampler_batch.py
