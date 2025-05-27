# Cuda memory

## Footprint of cudagraph

max_iteration doesn't change the footprint

asym = 4
peak mem
L = 6:  38MB
L = 12: 76MB
L = 14: ?

L = 12; asym = 4
self._MAX_ITERS_TO_CAPTURE = [400]: 75MB
self._MAX_ITERS_TO_CAPTURE = [400, 800, 1200]: 95MB

## New accurate measurements

### max_iteration change the footage very little

L = 6; asym = 4; bs = 2

Capturing CUDA graph for max_iter=800 (1/6)...
force_f CUDA Graph diff: 1204.00 MB

Capturing CUDA graph for max_iter=400 (2/6)...
force_f CUDA Graph diff: 1098.00 MB

Capturing CUDA graph for max_iter=200 (3/6)...
force_f CUDA Graph diff: 1052.00 MB

Capturing CUDA graph for max_iter=100 (4/6)...
force_f CUDA Graph diff: 1034.00 MB

Capturing CUDA graph for max_iter=40 (5/6)...
force_f CUDA Graph diff: 1020.00 MB

Capturing CUDA graph for max_iter=20 (6/6)...
force_f CUDA Graph diff: 1014.00 MB


### cuda graph mem is proportional to bs

L = 6; asym = 4
max_iter_se = 500; Nrv = 10

bs = 4
Capturing CUDA graph for max_iter=400 (1/2)...
force_f CUDA Graph diff: 2128.00 MB

Capturing CUDA graph for max_iter=200 (2/2)...
force_f CUDA Graph diff: 2062.00 MB

Initializing CUDA graph for get_fermion_obsr...
fermion_obsr CUDA Graph diff: 430.00 MB

bs = 3
Capturing CUDA graph for max_iter=400 (1/2)...
force_f CUDA Graph diff: 1620.00 MB

Capturing CUDA graph for max_iter=200 (2/2)...
force_f CUDA Graph diff: 1558.00 MB

Initializing CUDA graph for get_fermion_obsr...
fermion_obsr CUDA Graph diff: 324.00 MB

bs = 2
Initializing CUDA graph for force_f_fast...
Capturing CUDA graph for max_iter=400 (1/2)...
force_f CUDA Graph diff: 1120.00 MB

Capturing CUDA graph for max_iter=200 (2/2)...
force_f CUDA Graph diff: 1052.00 MB

Initializing CUDA graph for get_fermion_obsr...
fermion_obsr CUDA Graph diff: 216.00 MB


### cuda graph mem with size

bs = 2
aysm = 4
max_iter = 200

max_iter_se = 200
Nrv = 10

L = 6: 1463.75 MB
After setting precon: NVML Used: 1003.75 MB, diff: 24.00 MB

Initializing CUDA graph for force_f_fast...
Capturing CUDA graph for max_iter=200 (1/1)...
force_f CUDA Graph diff: 1076.00 MB

force_f CUDA graph initialization complete for batch sizes: [200]
After init force_f_graph: NVML Used: 2129.75 MB, diff: 1126.00 MB

Initializing CUDA graph for get_fermion_obsr...
fermion_obsr CUDA Graph diff: 88.00 MB

get_fermion_obsr CUDA graph initialization complete

After init se_graph: NVML Used: 2287.75 MB, diff: 158.00 MB

L = 12: 3623.75 MB
After setting precon: NVML Used: 1067.75 MB, diff: 68.00 MB

Initializing CUDA graph for force_f_fast...
Capturing CUDA graph for max_iter=200 (1/1)...
force_f CUDA Graph diff: 2084.00 MB

force_f CUDA graph initialization complete for batch sizes: [200]
After init force_f_graph: NVML Used: 3163.75 MB, diff: 2096.00 MB

Initializing CUDA graph for get_fermion_obsr...
fermion_obsr CUDA Graph diff: 88.00 MB

get_fermion_obsr CUDA graph initialization complete

After init se_graph: NVML Used: 3623.75 MB, diff: 460.00 MB

L = 18: 5683 MB
After setting precon: NVML Used: 1239.75 MB, diff: 240.00 MB

Initializing CUDA graph for force_f_fast...
Capturing CUDA graph for max_iter=200 (1/1)...
force_f CUDA Graph diff: 3096.00 MB

force_f CUDA graph initialization complete for batch sizes: [200]
After init force_f_graph: NVML Used: 4331.75 MB, diff: 3092.00 MB

Initializing CUDA graph for get_fermion_obsr...
fermion_obsr CUDA Graph diff: 104.00 MB

get_fermion_obsr CUDA graph initialization complete

After init se_graph: NVML Used: 5683.75 MB, diff: 1352.00 MB

L = 22: 7501 MB
After setting precon: NVML Used: 1433.75 MB, diff: 434.00 MB

Initializing CUDA graph for force_f_fast...
Capturing CUDA graph for max_iter=200 (1/1)...
force_f CUDA Graph diff: 3808.00 MB

force_f CUDA graph initialization complete for batch sizes: [200]
After init force_f_graph: NVML Used: 5091.75 MB, diff: 3658.00 MB

Initializing CUDA graph for get_fermion_obsr...
fermion_obsr CUDA Graph diff: 124.00 MB

get_fermion_obsr CUDA graph initialization complete

After init se_graph: NVML Used: 7501.75 MB, diff: 2410.00 MB


### cuda graph mem with Nrv
bs = 2
aysm = 4
max_iter = 200

max_iter_se = 200
Nrv = 10

L = 12: 3623.75 MB
After setting precon: NVML Used: 1067.75 MB, diff: 68.00 MB

Initializing CUDA graph for force_f_fast...
Capturing CUDA graph for max_iter=200 (1/1)...
force_f CUDA Graph diff: 2084.00 MB

force_f CUDA graph initialization complete for batch sizes: [200]
After init force_f_graph: NVML Used: 3163.75 MB, diff: 2096.00 MB

Initializing CUDA graph for get_fermion_obsr...
fermion_obsr CUDA Graph diff: 88.00 MB

get_fermion_obsr CUDA graph initialization complete

After init se_graph: NVML Used: 3623.75 MB, diff: 460.00 MB



