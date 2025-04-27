BLOCK_SIZE = (4, 4)

Lx = 6
bs = 1

21/9.5 = 2.2
bs = 5 / bs = 1

## Benchmark L=4
BLOCK_SIZE: (4, 4)
device: cuda
dtype: torch.float32
cdtype: torch.complex64
cg_cdtype: torch.complex64
debug_mode: False
J=1.0 
Nstep=6000 
Lx=4 
Ltau=160

bs = 1
  1%|█                                                                         | 89/6000 [02:20<2:40:00,  1.62s/it]

bs = 5
  0%|                                                                         | 10/6000 [01:08<11:06:11,  6.67s/it]


## Benchmark L=6
BLOCK_SIZE: (4, 4)
device: cuda
dtype: torch.float32
cdtype: torch.complex64
cg_cdtype: torch.complex64
debug_mode: False
J=1.0 
Nstep=6000 
Lx=6 
Ltau=240
cg_rtol: 1e-07 max_iter: 1000

bs: 5
  0%|                                                                          | 5/6000 [00:51<17:03:34, 10.24s/it]

bs: 1
  0%|▎                                                                         | 22/6000 [00:47<3:27:28,  2.08s/it]


## Benchmark L=8
device: cuda
dtype: torch.float32
cdtype: torch.complex64
cg_cdtype: torch.complex64
debug_mode: False
J=1.0 
Nstep=6000 
Lx=8 
Ltau=320
bs: 1
cg_rtol: 1e-07 max_iter: 1000

BLOCK_SIZE: (4, 4)
  0%|                                                                           | 8/6000 [00:24<4:55:07,  2.96s/it]

BLOCK_SIZE: (4, 8)
  0%|                                                                          | 10/6000 [00:29<4:51:26,  2.92s/it]
  
BLOCK_SIZE: (8, 4)
  0%|▏                                                                         | 18/6000 [00:54<4:53:29,  2.94s/it]

