#.bashrc

# torch 2.6.0+cu124

# pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu118



module load python3/3.9.3_anaconda2021.11_mamba  
module load cuda/12.0
module load matlab/R2023b
module load gcc/9.2.0
module load ninja/1.11.1-gcc-8.2.0-4drwoye

alias ll='ls -la'

export PATH=/common/software/install/manual/cuda/12.0/bin:$PATH
export LD_LIBRARY_PATH=/common/software/install/manual/cuda/12.0/lib64:$LD_LIBRARY_PATH


module load python/3.9.7
module load matlab/r2023b
module load gcc/9.2
module load cuda/12.3
export PATH=/share1/cuda/12.3/bin/:$PATH
export LD_LIBRARY_PATH=/share1/cuda/12.3/lib64:$LD_LIBRARY_PATH



ldd _C.cpython-39-x86_64-linux-gnu.so 
ldd: warning: you do not have execution permission for `./_C.cpython-39-x86_64-linux-gnu.so'
	linux-vdso.so.1 (0x00007fff111c2000)
	libc10.so => not found
	libtorch.so => not found
	libtorch_cpu.so => not found
	libtorch_python.so => not found
	libcudart.so.12 => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12 (0x000015114e656000)
	libc10_cuda.so => not found
	libtorch_cuda.so => not found
	libstdc++.so.6 => /share1/gcc/9.2.0/lib64/libstdc++.so.6 (0x000015114e277000)
	libm.so.6 => /usr/lib64/libm.so.6 (0x000015114def5000)
	libgcc_s.so.1 => /share1/gcc/9.2.0/lib64/libgcc_s.so.1 (0x000015114dcdd000)
	libpthread.so.0 => /usr/lib64/libpthread.so.0 (0x000015114dabd000)
	libc.so.6 => /usr/lib64/libc.so.6 (0x000015114d6e7000)
	/lib64/ld-linux-x86-64.so.2 (0x000015114e8fd000)
	libdl.so.2 => /usr/lib64/libdl.so.2 (0x000015114d4e3000)
	librt.so.1 => /usr/lib64/librt.so.1 (0x000015114d2db000)
