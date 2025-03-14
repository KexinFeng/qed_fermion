#!/bin/bash
#SBATCH --mail-type=END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fengx463@hku.hk     #    Email address to receive notification
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=./report/%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
#SBATCH --error=./report/%x_%j.err             #    Standard error log as $job_name_$job_id.errs

cd ${SLURM_SUBMIT_DIR}
module load python/3.9.7
python3 ../pbs_files/s.cmd
