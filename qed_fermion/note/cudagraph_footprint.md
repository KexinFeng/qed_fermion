# Footprint of cudagraph

max_iteration doesn't change the footprint

asym = 4
peak mem
L = 6:  38MB
L = 12: 76MB
L = 14: ?

L = 12; asym = 4
self._MAX_ITERS_TO_CAPTURE = [400]: 75MB
self._MAX_ITERS_TO_CAPTURE = [400, 800, 1200]: 95MB


# New more accurate measurements

## max_iteration change the footage very little

Capturing CUDA graph for max_iter=400 (1/2)...
force_f CUDA Graph diff: 2128.00 MB

Capturing CUDA graph for max_iter=200 (2/2)...
force_f CUDA Graph diff: 2062.00 MB


## cuda graph mem is proportional to bs

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


## cuda graph mem with size

aysm = 4
L = 6: 
L = 12:
L = 14: 
