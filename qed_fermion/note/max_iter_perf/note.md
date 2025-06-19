Here is a comparison of the three output files, focusing on the parameters that differ and are likely to impact throughput:

| Parameter                | report_cmp_large5_dtau2 (2135223) | report_cmp_large5_dtau2_sup1 (2146296) | report_cmp_large5_dtau2_sup2 (2147327) |
|--------------------------|:----------------------------------:|:--------------------------------------:|:--------------------------------------:|
| **SLURM_JOB_ID**         | 2135223                            | 2146296                                | 2147327                                |
| **SLURM_NODELIST**       | SPG-5-13                           | SPG-5-14                               | SPG-5-7                                |
| **max_iter_se**          | 100                                | 800                                    | 400                                    |
| **cg_rtol: max_iter**    | 1e-09: 1000                        | 1e-09: 1000                            | 1e-09: 400                             |
| **leapfrog_cmp_graph batch size** | [400]                   | [800]                                  | [400]                                  |
| **force_f_fast batch size**        | [400]                   | [800]                                  | [400]                                  |
| **force_f CUDA Graph diff**        | 3368.00 MB              | 3452.00 MB                             | 3368.00 MB                             |
| **fermion_obsr CUDA Graph diff**   | 444.00 MB               | 732.00 MB                              | 568.00 MB                              |
| **Execution time for init_stochastic_estimator** | 30.25 s | 192.10 s                               | 99.60 s                                |
| **After init se_graph: NVML Used** | 16839.69 MB             | 17211.69 MB                            | 16963.69 MB                            |
| **Step 0: Fermion computation took** | 9.95 s                | 63.58 s                                | 32.95 s                                |
| **Step 0: metropolis update took**   | 11.24 s                | 16.50 s                                | 11.23 s                                |

### Key Parameter Differences

- **max_iter_se** (stochastic estimator iterations):  
  - 100 (dtau2)  
  - 800 (dtau2_sup1)  
  - 400 (dtau2_sup2)  
  This is the main parameter that increases computation time per step as it increases.

- **cg_rtol: max_iter** (CG solver max iterations):  
  - 1000 (dtau2, dtau2_sup1)  
  - 400 (dtau2_sup2)  
  Lowering max_iter may speed up the CG solver but could affect convergence/accuracy.

- **leapfrog_cmp_graph and force_f_fast batch sizes**:  
  - [400] (dtau2, dtau2_sup2)  
  - [800] (dtau2_sup1)  
  Larger batch size increases memory usage and may slow down computation if the GPU is saturated.

- **CUDA Graph memory usage and initialization times**:  
  Both increase with higher max_iter_se and batch size.

### Summary

- The most significant difference affecting throughput is **max_iter_se** (100, 400, 800), which directly increases the time spent in the stochastic estimator and thus the total computation time per step.
- **Batch size** for CUDA graphs is doubled in dtau2_sup1 ([800] vs [400]), which also increases memory usage and may impact performance.
- **CG solver max_iter** is reduced in dtau2_sup2, which may speed up the solver but could affect results.
- All other parameters are the same or have negligible impact on throughput.

**In short:**  
- dtau2 (2135223) is the fastest (max_iter_se=100, batch=400, CG max_iter=1000)  
- dtau2_sup2 (2147327) is intermediate (max_iter_se=400, batch=400, CG max_iter=400)  
- dtau2_sup1 (2146296) is the slowest (max_iter_se=800, batch=800, CG max_iter=1000)  

The main driver of throughput difference is the value of **max_iter_se** and, to a lesser extent, the batch size and CG solver settings.