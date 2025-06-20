# Nrv tune

## L36
Here is the summary table for the Lx=36, Ltau=360 files with different `Nrv` values, including **Batch size for G**, **max_iter_se**, **max_iter**, **fermion computation time**, **metropolis update time**, and memory usage:

| Parameter                                | cL36a1J1.25K1_2149205<br>(Nrv=20) | cL36a1J1.25K1_2149201<br>(Nrv=30) | cL36a1J1.25K1_2148704<br>(Nrv=40) | cL36a1J1.25K1_2148708<br>(Nrv=80) |
|-------------------------------------------|:----------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:|
| **Nrv**                                  | 20                                 | 30                                | 40                                | 80                                |
| **max_iter_se**                          | 400                                | 400                               | 400                               | 400                               |
| **max_iter (CG)**                        | 400                                | 400                               | 400                               | 400                               |
| **Batch size for G**                     | 2                                  | 3                                 | 4                                 | 8                                 |
| **Step 0: Fermion computation (sec)**     | 5.66                               | 8.68                              | 11.75                             | 25.79                             |
| **Step 0: Metropolis update (sec)**       | 4.74                               | 4.73                              | 4.78                              | 4.67                              |
| **After init force_graph: NVML Used (MB)**| 2979.69                            | 2979.69                           | 2979.69                           | 2979.69                           |
| **After init force_graph: incr.by (MB)**  | 1834.00                            | 1834.00                           | 1834.00                           | 1834.00                           |
| **After init se_graph: NVML Used (MB)**   | 4785.69                            | 5583.69                           | 6395.69                           | 9593.69                           |
| **After init se_graph: incr.by (MB)**     | 1806.00                            | 2604.00                           | 3416.00                           | 6614.00                           |

---

**Notes:**
- **Batch size for G** is from "Batch size for G_delta_0_G_0_delta_ext".
- **max_iter_se** and **max_iter (CG)** are fixed at 400 for all runs.
- **Metropolis update time** and **force_graph memory** are nearly constant.
- **Fermion computation time** and **se_graph memory usage** increase significantly with `Nrv`.

**Trend:**  
As `Nrv` increases, both fermion computation time and memory usage after se_graph initialization grow rapidly, while other metrics remain stable.


## L40
Here is the updated summary table for the Lx=40, Ltau=400 files, now including **Batch size for G**, **max_iter_se**, and **max_iter**:

| Parameter                                | cL40a1J1.25K1_2149204<br>(Nrv=20) | cL40a1J1.25K1_2149200<br>(Nrv=30) | cL40a1J1.25K1_2148703<br>(Nrv=40) | cL40a1J1.25K1_2148707<br>(Nrv=80) |
|-------------------------------------------|:----------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:|
| **Nrv**                                  | 20                                 | 30                                | 40                                | 80                                |
| **max_iter_se**                          | 400                                | 400                               | 400                               | 400                               |
| **max_iter (CG)**                        | 400                                | 400                               | 400                               | 400                               |
| **Batch size for G**                     | 2                                  | 3                                 | 4                                 | 8                                 |
| **Step 0: Fermion computation (sec)**     | 8.28                               | 12.46                             | 16.89                             | 37.54                             |
| **Step 0: Metropolis update (sec)**       | 5.53                               | 5.61                              | 5.64                              | 5.61                              |
| **After init force_graph: NVML Used (MB)**| 3187.69                            | 3187.69                           | 3187.69                           | 3187.69                           |
| **After init force_graph: incr.by (MB)**  | 2042.00                            | 2042.00                           | 2042.00                           | 2042.00                           |
| **After init se_graph: NVML Used (MB)**   | 5547.69                            | 6653.69                           | 7739.69                           | 12113.69                          |
| **After init se_graph: incr.by (MB)**     | 2360.00                            | 3466.00                           | 4552.00                           | 8926.00                           |

---

**Notes:**
- **Batch size for G** refers to the "Batch size for G_delta_0_G_0_delta_ext" entries.
- **max_iter_se** is the stochastic estimator iteration count.
- **max_iter (CG)** is the conjugate gradient solver's maximum iterations.

**Trends:**  
- Increasing `Nrv` increases **Batch size for G**, **fermion computation time**, and **se_graph memory usage**.
- **Metropolis update time** and **force_graph memory** remain nearly constant.
- **max_iter_se** and **max_iter (CG)** are fixed at 400 for all runs.

## L46

Here is the summary table for the Lx=46, Ltau=460 files with different `Nrv` values, including **Batch size for G**, **max_iter_se**, **max_iter**, **fermion computation time**, **metropolis update time**, and memory usage:

| Parameter                                | cL46a1J1.25K1_2149203<br>(Nrv=20) | cL46a1J1.25K1_2149199<br>(Nrv=30) | cL46a1J1.25K1_2148702<br>(Nrv=40) | cL46a1J1.25K1_2148706<br>(Nrv=80) |
|-------------------------------------------|:----------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:|
| **Nrv**                                  | 20                                 | 30                                | 40                                | 80                                |
| **max_iter_se**                          | 400                                | 400                               | 400                               | 400                               |
| **max_iter (CG)**                        | 400                                | 400                               | 400                               | 400                               |
| **Batch size for G**                     | 2                                  | 3                                 | 4                                 | 8                                 |
| **Step 0: Fermion computation (sec)**     | 15.38                              | 23.29                             | 32.15                             | 70.42                             |
| **Step 0: Metropolis update (sec)**       | 9.00                               | 9.08                              | 9.12                              | 8.99                              |
| **After init force_graph: NVML Used (MB)**| 3547.69                            | 3547.69                           | 3547.69                           | 3547.69                           |
| **After init force_graph: incr.by (MB)**  | 2402.00                            | 2402.00                           | 2402.00                           | 2402.00                           |
| **After init se_graph: NVML Used (MB)**   | 7349.69                            | 9149.69                           | 10963.69                          | 18197.69                          |
| **After init se_graph: incr.by (MB)**     | 3802.00                            | 5602.00                           | 7416.00                           | 14650.00                          |

---

**Notes:**
- **Batch size for G** is from "Batch size for G_delta_0_G_0_delta_ext".
- **max_iter_se** and **max_iter (CG)** are fixed at 400 for all runs.
- **Metropolis update time** and **force_graph memory** are nearly constant.
- **Fermion computation time** and **se_graph memory usage** increase significantly with `Nrv`.

**Trend:**  
As `Nrv` increases, both fermion computation time and memory usage after se_graph initialization grow rapidly, while other metrics remain stable.


##L50
Here is the summary table for the Lx=50, Ltau=500 files with different `Nrv` values, including **Batch size for G**, **max_iter_se**, **max_iter**, **fermion computation time**, **metropolis update time**, and memory usage:

| Parameter                                | cL50a1J1.25K1_2149202<br>(Nrv=20) | cL50a1J1.25K1_2149198<br>(Nrv=30) | cL50a1J1.25K1_2148701<br>(Nrv=40) | cL50a1J1.25K1_2148705<br>(Nrv=80) |
|-------------------------------------------|:----------------------------------:|:---------------------------------:|:---------------------------------:|:---------------------------------:|
| **Nrv**                                  | 20                                 | 30                                | 40                                | 80                                |
| **max_iter_se**                          | 400                                | 400                               | 400                               | 400                               |
| **max_iter (CG)**                        | 400                                | 400                               | 400                               | 400                               |
| **Batch size for G**                     | 2                                  | 3                                 | 4                                 | 8                                 |
| **Step 0: Fermion computation (sec)**     | 20.71                              | 31.19                             | 42.25                             | 91.91                             |
| **Step 0: Metropolis update (sec)**       | 11.84                              | 11.79                             | 11.79                             | 11.82                             |
| **After init force_graph: NVML Used (MB)**| 3791.69                            | 3791.69                           | 3791.69                           | 3791.69                           |
| **After init force_graph: incr.by (MB)**  | 2642.00                            | 2642.00                           | 2642.00                           | 2642.00                           |
| **After init se_graph: NVML Used (MB)**   | 8207.69                            | 10337.69                          | 12435.69                          | 20901.69                          |
| **After init se_graph: incr.by (MB)**     | 4416.00                            | 6546.00                           | 8644.00                           | 17110.00                          |

---

**Notes:**
- **Batch size for G** is from "Batch size for G_delta_0_G_0_delta_ext".
- **max_iter_se** and **max_iter (CG)** are fixed at 400 for all runs.
- **Metropolis update time** and **force_graph memory** are nearly constant.
- **Fermion computation time** and **se_graph memory usage** increase significantly with `Nrv`.

**Trend:**  
As `Nrv` increases, both fermion computation time and memory usage after se_graph initialization grow rapidly, while other metrics remain stable.



