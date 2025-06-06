# Performance of Stochastic Estimator

## Setup
Lx = 20
Ltau = 800

max_tau_block_idx: 10
N_leapfrog = 3

### Natural batching, Nrv=100
Batch_Size = Nrv = 100

Step 1: metropolis update took 26430.83 ms
Step 1: Fermion computation took 15796.82 ms

Before init se_graph: NVML Used: 4685.75 MB
Inside capture start_mem_0: 4909.75 MB

Inside capture start_mem: 12045.75 MB

Inside capture end_mem: 12397.75 MB

fermion_obsr CUDA Graph diff: 352.00 MB

get_fermion_obsr CUDA graph initialization complete
After init se_graph: NVML Used: 12397.75 MB, incr.by: 7712.00 MB


Execution time for init_stochastic_estimator: 48.60 seconds



### More batches, Nrv=100
Batch_Size = Nrv//10 = 10

Step 1: metropolis update took 26400 ms
Step 1: Fermion computation took 17280 ms


Before init se_graph: NVML Used: 4685.75 MB
Inside capture start_mem_0: 4909.75 MB

Inside capture start_mem: 11065.75 MB

Inside capture end_mem: 10605.75 MB

fermion_obsr CUDA Graph diff: -460.00 MB

get_fermion_obsr CUDA graph initialization complete
After init se_graph: NVML Used: 10605.75 MB, incr.by: 5920.00 MB


Execution time for init_stochastic_estimator: 57.28 seconds



### Natural batching, Nrv=50
Batch_Size = Nrv = 50

Step 1: metropolis update took 26.40 sec
Step 1: Fermion computation took 5.98 sec


Before init se_graph: NVML Used: 4685.75 MB
Inside capture start_mem_0: 4787.75 MB

Inside capture start_mem: 8385.75 MB

Inside capture end_mem: 8605.75 MB

fermion_obsr CUDA Graph diff: 220.00 MB

get_fermion_obsr CUDA graph initialization complete

Execution time for init_stochastic_estimator: 18.71 seconds

After init se_graph: NVML Used: 8605.75 MB, incr.by: 3920.00 MB



### Natural batching, Nrv=10
Batch_Size = Nrv = 10

Step 1: metropolis update took 26.40 sec
Step 1: Fermion computation took 0.66 sec

Before init se_graph: NVML Used: 4685.75 MB
Inside capture start_mem_0: 4663.75 MB

Inside capture start_mem: 5419.75 MB

Inside capture end_mem: 5533.75 MB

fermion_obsr CUDA Graph diff: 114.00 MB

get_fermion_obsr CUDA graph initialization complete
After init se_graph: NVML Used: 5533.75 MB, incr.by: 848.00 MB


Execution time for init_stochastic_estimator: 2.67 seconds

