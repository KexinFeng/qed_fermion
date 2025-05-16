# Performance of cudagraph

Here cudagraph refers to force_f_fast cudagraph.

bs = 2 for all

turn on cuda_graph
L = 8
asym = 2
1.4 it/s

turn off cuda_graph
L = 8
asym = 2
5.33 s/it

turn off cuda_graph
L = 6
asym = 4
7.87 s/it

turn on cuda_graph
L = 6
asym = 4
1.05 it/s
