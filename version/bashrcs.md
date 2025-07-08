# Note of environment

## agate
```bashrc
# .bashrc startup script
# This is sourced by .bash_profile on login shells
# It is sourced by the bash shell it's self for interactive non-login shells (i.e. jobs)
# If this has already been sourced
if ${HOMEBASHSOURCED:-false} ;  then
  return 0 # Do nothing AND immediately exit so the rest of the file is not loaded
fi

# If it has not been sourced
# Set the source flag to true
HOMEBASHSOURCED=true

# Source the global system definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# Set your umask so only you and your group members can read/execute your files
umask 027

# Set the shell prompt.
PS1="\u@\h [\w] % "

# Add your aliases here.
# alias s='ssh -X'

# Set your environment variables here.
# export VISUAL=vim

# torch 2.6.0+cu124

# pip install --upgrade pip
# pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu118 [--target /users/4/fengx463/.local/lib/python3.9/site-packages/]
## pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
# export CUDA_HOME=/common/software/install/migrated/cuda/11.2
# export LD_LIBRARY_PATH=/common/software/install/manual/cuda/12.0/lib64:$LD_LIBRARY_PATH 

# Load modules here
module load python3/3.9.3_anaconda2021.11_mamba  
module load cuda/12.0
# module load cuda/11.2
module load matlab/R2023b
module load gcc/9.2.0
module load ninja/1.11.1-gcc-8.2.0-4drwoye

alias ll='ls -la'

export PATH=/common/software/install/manual/cuda/12.0/bin:$PATH
export LD_LIBRARY_PATH=/common/software/install/manual/cuda/12.0/lib64:$LD_LIBRARY_PATH
```


## hpc2021
