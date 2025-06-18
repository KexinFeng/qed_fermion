# Performance of cudagraph

Here cudagraph refers to force_f_fast cudagraph.

bs = 2 for all

turn on cuda_graph
L = 8
asym = 2
1.4 it/s (1.39 it/s w/o inference mode)

turn off cuda_graph
L = 8
asym = 2
5.33 s/it (5.80 s/it w/o inference mode)


turn on cuda_graph
L = 6
asym = 4
1.05 it/s (1.01 it/s w/o inference mode)

turn off cuda_graph
L = 6
asym = 4
7.87 s/it (8.68 s/it w/o inference mode)
