# Performance

## Setup
dtype = torch.float64
cdtype = torch.complex128
cg_dtype = torch.complex128

```py
    J = float(os.getenv("J", '1.0'))
    Nstep = int(os.getenv("Nstep", '6000'))
    Lx = int(os.getenv("Lx", '10'))
    Ltau = int(os.getenv("Ltau", '400'))
    print(f'J={J} \nNstep={Nstep}')

    hmc = HmcSampler(Lx=Lx, Ltau=Ltau, J=J, Nstep=Nstep)
```

*M4 Pro cpu*
  0%|                                      | 10/6000 [04:00<39:51:17, 23.95s/it]

*MSI cpu*
  0%|                                      | 1/6000 [02:21<235:09:53, 141.12s/it]

*MSI l40s*
  0%|                                       | 6/6000 [06:35<109:44:04, 65.91s/it]


### CG convergence iteration

self.cg_rtol = 1e-7
precon = pi_flux_precon
    iter = 20; % 20
    thrhld = 0.1; % 0.1
    diagcomp = 0.05; % 0.05

disp(sprintf('%.2g, %d, %d', 1 - double(nnz(O_inv1)) / double(Nx * Ny), nnz(O_inv1), Nx * Ny));
Sparsity of O_inv1:
0.9, 160220000, 1600000000
Sparsity of O_inv::
0.99967713, 516600, 1600000000

Average iteration: 48


## Setup2

dtype = torch.float32
cdtype = torch.complex64
cg_dtype = torch.complex64


Average CG convergence iteration: 48

M4 Pro cpu
  0%|                                         | 9/6000 [03:33<39:37:50, 23.81s/it]

