#!/bin/bash
#SBATCH --mail-type=END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fengx463@hku.hk     #    Email address to receive notification
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=./report/%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
#SBATCH --error=./report/%x_%j.err             #    Standard error log as $job_name_$job_id.errs
#SBATCH --gres=gpu:1

cd ${SLURM_SUBMIT_DIR}
module load python/3.9.7

# Check CUDA version from driver
nvidia-smi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d '.' -f1-2)
echo $CUDA_VERSION

# Map CUDA version to PyTorch compatible version
case $CUDA_VERSION in
  "12.0") PYTORCH_CUDA="cu121" ;;
  "11.8") PYTORCH_CUDA="cu118" ;;
  "11.7") PYTORCH_CUDA="cu117" ;;
  *) echo "Unsupported CUDA version"; exit 1 ;;
esac

# Install PyTorch with compatible CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# check install result
python3 -c "import\ torch\;\ print\(torch.cuda.is_available\(\)\)"

