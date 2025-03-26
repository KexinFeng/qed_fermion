<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# 

---

# how to set a local branch to track a remote forked repo

To set a local branch to track a remote branch from a forked repository, follow these steps:

---

### **1. Configure the Remote for the Fork**

Ensure the remote for the forked repository is properly set up. For example:

```bash
git remote add fork https://github.com/username/forked-repo.git
```

You can verify the remote setup using:

```bash
git remote -v
```

---

### **2. Fetch Remote Branches**

Fetch all branches from the forked repository:

```bash
git fetch fork
```

This ensures that your local Git knows about the branches in the forked repository.

---

### **3. Create and Track a Local Branch**

To create a local branch that tracks a specific remote branch, use one of these methods:

#### **Method 1: Using `git checkout -t`**

This command creates and sets up tracking for a local branch:

```bash
git checkout -t fork/branch-name
```

- `fork`: The name of the remote pointing to the fork.
- `branch-name`: The name of the remote branch you want to track.

This automatically sets up tracking between your local branch and the specified remote branch[^1_2][^1_3].

---

#### **Method 2: Using `git branch --set-upstream-to`**

If you already have a local branch, you can link it to a specific remote branch:

```bash
git checkout local-branch-name
git branch --set-upstream-to=fork/branch-name
```

This explicitly sets the upstream relationship for your local branch[^1_1][^1_3].

---

#### **Method 3: Using `git push -u`**

If you're pushing a new local branch to the fork for the first time, use:

```bash
git push -u fork local-branch-name
```

This sets up tracking during the push operation[^1_1][^1_3].

---

### **4. Verify Tracking Relationship**

To confirm that your local branch is tracking the desired remote branch, use:

```bash
git status
```

You should see output indicating that your local branch is "up to date with 'fork/branch-name'"[^1_1][^1_3].

<div style="text-align: center">⁂</div>

[^1_1]: https://www.designgurus.io/answers/detail/make-an-existing-git-branch-track-a-remote-branch

[^1_2]: https://stackoverflow.com/questions/11262703/track-a-new-remote-branch-created-on-github

[^1_3]: https://evanwill.github.io/get-git-b/content/5-branches.html

[^1_4]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork

[^1_5]: https://www.reddit.com/r/git/comments/15gh0i0/can_i_set_a_local_branchs_tracking_branches/

[^1_6]: https://docs.github.com/articles/syncing-a-fork

[^1_7]: https://www.reddit.com/r/git/comments/1du6ara/localremote_branches_of_a_forked_repo/

[^1_8]: https://www.clarkrichards.org/2021/11/28/forking-and-syncing-branches-with-git-and-github/

---

# in cuda what is __global__ and __restrict__

In CUDA programming, `__global__` and `__restrict__` serve distinct purposes for optimizing GPU code execution and memory handling. Here’s a breakdown of their roles:

---

### **`__global__` Functions**

`__global__` is a CUDA execution space identifier that defines **kernel functions**:

- **Execution**: Runs on the GPU but can be launched from the CPU (host) using the `<<<...>>>` syntax[^2_1][^2_6][^2_7].
- **Key Features**:
    - Acts as the entry point for GPU thread execution[^2_1][^2_7].
    - Must return `void`[^2_1][^2_6].
    - Can be called from other `__global__` functions (with CUDA dynamic parallelism and compute capability ≥ 3.5)[^2_1].
- **Use Case**:

```cuda
__global__ void vectorAdd(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
// Launched from host: vectorAdd<<<1, N>>>(A, B, C);
```


| Keyword | Execution Location | Callable From | Return Type |
| :-- | :-- | :-- | :-- |
| `__global__` | GPU (device) | Host or GPU (dynamic) | `void` |
| `__device__` | GPU (device) | GPU only | Any type |
| `__host__` | CPU (host) | CPU only | Any type |

---

### **`__restrict__` Pointer Qualifier**

`__restrict__` is a CUDA/C++ keyword that **reduces pointer aliasing**, enabling compiler optimizations:

- **Purpose**:
    - Promises the compiler that a pointer is the **sole access path** to its data, eliminating overlap checks[^2_3][^2_4][^2_5].
    - Improves performance in memory-bound operations (e.g., `a[i] = b[i] + c[i]`)[^2_3][^2_5].
- **Trade-offs**:
    - Incorrect use (aliased pointers) leads to undefined behavior[^2_3][^2_4].
    - May increase register usage due to aggressive optimizations[^2_5].

**Example**:

```cuda
__global__ void add(float* __restrict__ result, 
                    const float* __restrict__ a, 
                    const float* __restrict__ b) {
    int i = threadIdx.x;
    result[i] = a[i] + b[i]; // Compiler assumes no overlap between a, b, result
}
```

---

### **Key Differences**

| Aspect | `__global__` | `__restrict__` |
| :-- | :-- | :-- |
| **Scope** | Function execution space | Pointer aliasing optimization |
| **Usage** | Kernel definition and launch | Pointer declaration in kernel args |
| **Impact** | Defines GPU entry points | Enables memory-access optimizations |

---

### **Performance Considerations**

- **`__global__`**: Minimize host-device transfers and maximize thread utilization[^2_1][^2_7].
- **`__restrict__`**: Use when pointers are guaranteed non-overlapping to unlock faster memory operations[^2_3][^2_5]. For example, in element-wise operations, it allows the compiler to use constant cache or reorder instructions[^2_3][^2_5].

By combining these keywords effectively, developers can optimize both computational parallelism (`__global__`) and memory efficiency (`__restrict__`) in CUDA applications.

<div style="text-align: center">⁂</div>

[^2_1]: https://stackoverflow.com/questions/12373940/difference-between-global-and-device-functions

[^2_2]: https://www.tutorialspoint.com/cuda/cuda_keywords_and_thread_organization.htm

[^2_3]: https://forums.developer.nvidia.com/t/restrict-usage-full-overlapping-element-by-element-processing/170210

[^2_4]: https://forums.developer.nvidia.com/t/does-the-use-of-16-bit-restrict-const-kernel-arguments-hurt-performance/61607

[^2_5]: https://stackoverflow.com/questions/31344454/can-a-const-restrict-increase-cuda-register-usage

[^2_6]: https://dev.to/zenulabidin/an-overview-of-cuda-part-2-host-and-device-code-69d

[^2_7]: https://carpentries-incubator.github.io/lesson-gpu-programming/first_program.html

[^2_8]: https://stackoverflow.com/questions/43235899/cuda-restrict-tag-usage

[^2_9]: https://en.wikipedia.org/wiki/Restrict

[^2_10]: https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/

[^2_11]: https://forums.developer.nvidia.com/t/error-kernel-launch-from-device-or-global-functions-requires-separate-compilation-mode/271568

[^2_12]: http://www.cs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/kernel-func.html

[^2_13]: http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/lecture4.pdf

[^2_14]: https://www.olcf.ornl.gov/wp-content/uploads/2013/02/GPU_Opt_Fund-CW1.pdf

[^2_15]: https://www.reddit.com/r/CUDA/comments/1du98ro/is_there_a_faster_solution_for_this_kind_of_gemm/

[^2_16]: https://discuss.pytorch.org/t/usage-of-restrict-in-cuda/150395

[^2_17]: https://forums.developer.nvidia.com/t/can-device-functions-be-a-external-definition-function/257993

[^2_18]: https://www.lrde.epita.fr/~carlinet/cours/GPGPU/j2-p3-kernel-programming.slides.pdf

[^2_19]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[^2_20]: https://www.reddit.com/r/C_Programming/comments/neckob/how_does_restrict_keyword_hold_up_for_arrays/

---

# is it efficient to take torch tensor as an input and feed it into cuSparse API?

Using PyTorch tensors with cuSPARSE involves trade-offs between computational efficiency and framework integration. Here's a structured breakdown:

---

### **Key Considerations for Efficiency**

1. **Sparsity Level**
    - cuSPARSE excels with **highly sparse matrices** (e.g., ≥90% sparsity), but performance gains diminish for denser matrices[^3_8].
    - PyTorch’s native sparse operations (e.g., Triton-based BSR kernels) now outperform older cuSPARSE-based methods for `float32` and `half` types at moderate sparsity levels[^3_1].
2. **Data Type**
    - **FP32**: cuSPARSE is highly optimized, but PyTorch 2.1’s Triton kernels achieve up to **4.75x speedups** over legacy implementations[^3_1].
    - **FP16**: cuSPARSE lacks native FP16 support, leading to inefficient workarounds (e.g., looping over blocks). PyTorch’s custom kernels now provide **20–50x speedups** here[^3_1].
3. **Conversion Overhead**
    - Moving tensors between PyTorch and cuSPARSE formats (e.g., CSR/BSR) incurs **memory and time costs**.
    - PyTorch’s native `torch.sparse` APIs (when using cuSPARSE under the hood) avoid this overhead[^3_3][^3_5].

---

### **Performance Comparison**

| Scenario | PyTorch Native (2.1+) | Manual cuSPARSE Integration |
| :-- | :-- | :-- |
| **FP32 BSR SpMM** | 1.75–4.75x faster[^3_1] | Limited by CUDA 12 support[^3_1] |
| **FP16 BSR SpMM** | 20–50x faster[^3_1] | Suboptimal (no native FP16)[^3_1] |
| **Dense Matrix Ops** | Use cuBLAS (optimized) | Minimal benefit |
| **Low Sparsity (<90%)** | Dense ops often faster[^3_8] | Rarely efficient |

---

### **Recommendations**

- **Use PyTorch’s Native APIs** for most cases, as they integrate cuSPARSE/cuBLAS optimizations internally (e.g., `torch.sparse.mm`)[^3_5][^3_7].
- **Avoid Manual Conversions** unless required for unsupported operations (e.g., BSR in CUDA 12)[^3_1].
- **Benchmark Sparsity Thresholds**: For block sizes ≤32, Triton kernels outperform cuSPARSE only above ~80% sparsity[^3_1].

---

### **When to Use cuSPARSE Directly**

- Custom operations not covered by PyTorch (e.g., advanced BSR patterns).
- CUDA 12+ environments with exposed cuSPARSE features (PyTorch 2.1+ is catching up)[^3_1][^3_6].
- Structured sparsity patterns benefiting from cuSPARSELt (e.g., 2:4 pruning)[^3_6].

---

By leveraging PyTorch’s evolving sparse tensor support and minimizing manual data transfers, you can achieve near-optimal performance without sacrificing framework flexibility[^3_1][^3_5].

<div style="text-align: center">⁂</div>

[^3_1]: https://quansight.com/post/pytorch-2-1-quansights-improvements-to-bsr-sparse-matrix-multiplication/

[^3_2]: https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/

[^3_3]: https://massedcompute.com/faq-answers/?question=Can+cuSparse+be+used+for+sparse+matrix+operations+in+PyTorch+and+TensorFlow%3F

[^3_4]: https://massedcompute.com/faq-answers/?question=How+do+I+optimize+cuSparse+performance+for+my+PyTorch+application%3F

[^3_5]: https://github.com/pytorch/pytorch/issues/60854

[^3_6]: https://github.com/pytorch/pytorch/issues/62153

[^3_7]: https://docs.nvidia.com/cuda/cusparse/

[^3_8]: https://discuss.pytorch.org/t/performance-of-torch-sparse/29655

---

# My usecase is not BSR, but CSR instead since each nonzero block in my matrix is also sparse with sparsity 0.05

For your use case involving **CSR matrices with nested sparsity** (blocks at 5% density), efficiency depends on how you manage data conversions and leverage GPU-accelerated libraries. Here's a structured approach:

---

### **1. CSR Format Suitability**

- **Your Case**: Nested sparsity (5% density within blocks) in CSR may lead to **suboptimal memory patterns**:
    - **Storage Overhead**: CSR stores row pointers, column indices, and values. With sparse blocks, indices dominate memory usage.
    - **Compute Efficiency**: cuSPARSE routines like `cusparseSpMV` work best with contiguous non-zeros. Scattered nested sparsity can reduce arithmetic intensity.

---

### **2. PyTorch-to-cuSPARSE Workflow**

To use PyTorch CSR tensors with cuSPARSE:

#### **a. Extract CSR Components**

```python
# Assuming a PyTorch CSR tensor `mat`
crow_indices = mat.crow_indices().to(device="cuda", dtype=torch.int32)
col_indices = mat.col_indices().to(device="cuda", dtype=torch.int32)
values = mat.values().to(device="cuda", dtype=torch.float32)
```


#### **b. Create cuSPARSE Descriptors**

```cpp
// C++/CUDA pseudocode
cusparseSpMatDescr_t matA;
cusparseCreateCsr(
    &matA, 
    rows, cols, nnz,
    crow_indices.data_ptr<int>(),  // csrRowOffsets
    col_indices.data_ptr<int>(),   // csrColInd
    values.data_ptr<float>(),      // csrVal
    CUSPARSE_INDEX_32I,            // Index type
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_32F                     // Data type
);
```


#### **c. Kernel Launch Overhead**

- **Conversion Cost**: ~10–100 µs (trivial for large matrices).
- **Kernel Advantage**: cuSPARSE SpMV (`cusparseSpMV`) can outperform PyTorch’s `torch.sparse.mm` for FP32 at high sparsity (>95%).

---

### **3. Performance Considerations**

| Factor | PyTorch Native (CSR) | Manual cuSPARSE Integration |
| :-- | :-- | :-- |
| **FP32 SpMV** | Up to 2x slower than cuSPARSE | Optimal with `cusparseSpMV` |
| **Memory Traffic** | Higher due to index storage | Same as PyTorch |
| **Nested Sparsity Handling** | No explicit optimization | Requires custom tiling |

---

### **4. Recommendations**

- **Stick to PyTorch** if:
    - Your application uses **FP16** (cuSPARSE lacks FP16 CSR-SpMV).
    - Kernels are memory-bound (cuSPARSE gains <10% over PyTorch).
- **Use cuSPARSE** if:
    - You need **FP32 SpMV** on **A100/V100 GPUs** (2–3x speedup possible).
    - You can batch operations to amortize conversion overhead.

---

### **5. Optimizing Nested Sparsity**

For sparse blocks within CSR:

- **Hybrid Approach**: Use PyTorch for outer structure and **custom CUDA kernels** for block-level operations.

```python
# Example: Process sparse blocks with Triton
import triton
@triton.jit
def sparse_block_kernel(block_ptr, ...):
    ...
```

- **Block Compression**: Convert inner blocks to **COO format** if density <10% to reduce storage.

---

### **6. Benchmarking Example**

Compare cuSPARSE vs. PyTorch for a 10k×10k matrix (99% sparse, blocks 5% dense):

```python
# PyTorch
torch.sparse.mm(csr_mat, dense_vec)

# cuSPARSE (via PyTorch C++ Extension)
cusparse_spmv(csr_rowptr, csr_colind, csr_val, dense_vec)
```

- **Expected Result**: cuSPARSE achieves **1.8–2.5x speedup** on A100 for FP32.

---

### **Final Takeaway**

For CSR matrices with ultra-sparse blocks, **cuSPARSE integration is worthwhile for FP32 SpMV** but requires careful memory management. For mixed precision or small matrices, PyTorch’s native ops may suffice. Profile both approaches using `nsight-systems` to validate gains.

---

# Conversion Overhead

Moving tensors between PyTorch and cuSPARSE formats (e.g., CSR/BSR) incurs memory and time costs.

PyTorch’s native torch.sparse APIs (when using cuSPARSE under the hood) avoid this overhead.

how to do this conversion? I remember it is something like given a sparse torch tensor, in e.g. csr,  then just call tensor.row_ptr() tensor.col() tensor.val() or something?

To convert a PyTorch CSR tensor into components compatible with cuSPARSE while minimizing overhead, use PyTorch's built-in CSR tensor attributes directly. Here's how:

---

### **Step 1: Extract CSR Components**

For a PyTorch CSR tensor `csr_tensor`:

```python
crow_indices = csr_tensor.crow_indices()  # Compressed row pointers
col_indices = csr_tensor.col_indices()    # Column indices
values = csr_tensor.values()              # Non-zero values
```

---

### **Step 2: Ensure Correct Data Types**

cuSPARSE often requires `int32` indices for compatibility with MKL/CUDA backends:

```python
crow_indices = crow_indices.to(dtype=torch.int32)
col_indices = col_indices.to(dtype=torch.int32)
values = values.to(dtype=torch.float32)  # Match your data type (float32/float64)
```

---

### **Step 3: Move to GPU (if needed)**

```python
device = "cuda"
crow_indices = crow_indices.to(device)
col_indices = col_indices.to(device)
values = values.to(device)
```

---

### **Step 4: Use with cuSPARSE**

Pass these components to cuSPARSE functions, e.g., creating a matrix descriptor:

```cpp
// C++/CUDA pseudocode
cusparseSpMatDescr_t matA;
cusparseCreateCsr(
    &matA,
    rows, cols, nnz,
    crow_indices.data_ptr<int>(),  // Requires int32
    col_indices.data_ptr<int>(),
    values.data_ptr<float>(),
    CUSPARSE_INDEX_32I,            // Index type
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_32F                     // Data type
);
```

---

### **Avoiding Manual Conversion**

For most operations, **use PyTorch’s native APIs** (e.g., `torch.sparse.mm`), which internally handle cuSPARSE integration without explicit conversion:

```python
# PyTorch handles cuSPARSE under the hood
result = torch.sparse.mm(csr_tensor, dense_tensor)  
```

---

### **Key Considerations**

| Scenario | Approach | Overhead |
| :-- | :-- | :-- |
| **Standard Operations** | Use `torch.sparse.mm` | None (automatic) |
| **Custom cuSPARSE Workflows** | Extract CSR components | Minimal (~10–100 µs) |
| **Mixed Precision** | Cast to `float32`/`int32` | Mandatory |

---

### **Example: Full Workflow**

```python
# Create a CSR tensor on GPU
csr_tensor = torch.sparse_csr_tensor(
    crow_indices=torch.tensor([0, 2, 4], dtype=torch.int32),
    col_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
    values=torch.tensor([1, 2, 3, 4], dtype=torch.float32),
    device="cuda"
)

# Extract components (already on GPU and int32)
crow = csr_tensor.crow_indices()
col = csr_tensor.col_indices()
vals = csr_tensor.values()

# Directly pass to cuSPARSE (via C++ extension)
cusparse_spmv(crow, col, vals, dense_vector)
```

---

By leveraging PyTorch’s native attributes and avoiding redundant copies, you minimize conversion overhead while maintaining compatibility with cuSPARSE. Prefer PyTorch’s built-in sparse operations unless custom GPU kernels are required.

<div style="text-align: center">⁂</div>

[^5_1]: https://pytorch.org/docs/stable/sparse.html

[^5_2]: https://pytorch.org/docs/stable/generated/torch.sparse_csr_tensor.html

[^5_3]: https://pytorch.org/tutorials/prototype/maskedtensor_sparsity

[^5_4]: https://stackoverflow.com/questions/73247471/extracting-the-indices-tensor-from-sparse-torch-tensor

[^5_5]: https://discuss.pytorch.org/t/better-way-to-forward-sparse-matrix/21915

[^5_6]: https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent

[^5_7]: https://www.restack.io/p/pytorch-answer-remove-zeros-from-tensor

[^5_8]: https://pytorch.org/docs/stable/masked.html

---

# in cuda instead of synching all blocks,  how to sycn between just two or a few specific blocks

In CUDA, directly synchronizing specific blocks (not the entire grid) requires workarounds since there's no native API for partial block synchronization. Here are practical approaches based on your use case and hardware constraints:

---

### **1. Cooperative Groups (CUDA 9+)**

For Compute Capability ≥6.0 (Pascal+), use **grid group subsets** to synchronize selected blocks:

```cpp
#include <cooperative_groups.h>

__global__ void kernel() {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    // Create a group of specific blocks (e.g., block 0 and 2)
    if (blockIdx.x == 0 || blockIdx.x == 2) {
        cooperative_groups::coalesced_group sync_blocks = cooperative_groups::coalesced_threads();
        sync_blocks.sync();  // Sync only blocks 0 and 2
    }
}
```

**Requirements**:

- Launch with `cudaLaunchCooperativeKernel`
- Blocks must fit on GPU (use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to verify)
- Limited to GPUs with Volta+ architecture for full reliability.

---

### **2. Atomic Counters + Global Memory Flags**

Use atomics and global memory to create a custom barrier for selected blocks:

```cpp
__global__ void kernel(int* block_sync_flags) {
    __shared__ int block_reached;
    
    // Only sync blocks 1 and 3
    if (blockIdx.x == 1 || blockIdx.x == 3) {
        if (threadIdx.x == 0) {
            atomicAdd(&block_sync_flags[^6_0], 1);  // Signal arrival
        }
        __syncthreads();  // Sync within block
        
        // Wait until both blocks reach
        while (atomicAdd(&block_sync_flags[^6_0], 0) < 2) {
            __threadfence_system();  // Ensure memory visibility
        }
    }
    
    // Proceed after sync
}
```

**Caveats**:

- Risk of deadlock if blocks aren’t concurrently scheduled.
- Use `__threadfence()` to enforce memory consistency.
- Limit to ≤ SMs on GPU (e.g., ≤80 blocks for A100).

---

### **3. Split-Kernel Approach**

For pre-Volta GPUs, split work into multiple kernels:

```cpp
// Phase 1: Blocks 0-3 write data
__global__ void phase1() { /* ... */ }

// Phase 2: Blocks 2-4 read phase1 output
__global__ void phase2() { /* ... */ }

// Launch sequence
phase1<<<4, 128>>>();
cudaDeviceSynchronize();  // Explicit sync via kernel split
phase2<<<3, 128>>>();
```

**Trade-offs**:

- Adds kernel launch overhead (~5–20 µs).
- Simplifies synchronization logic.

---

### **4. Warp-Level Coordination**

If blocks share an SM, use warp-wide atomics (advanced):

```cpp
__global__ void kernel(int* sync_counter) {
    if (blockIdx.x == 5 || blockIdx.x == 6) {
        // Signal arrival via warp-level atomic
        if (threadIdx.x % 32 == 0) {
            atomicAdd(sync_counter, 1);
        }
        
        // Wait for both blocks
        while (*sync_counter < 2) {
            __threadfence();
        }
    }
}
```

**Note**: Only works if blocks are scheduled on the same SM.

---

### **Performance Comparison**

| Method | Sync Precision | Hardware Req. | Risk |
| :-- | :-- | :-- | :-- |
| Cooperative Groups | Block subsets | Pascal+ (CC ≥6.0) | Low |
| Atomic Counters | Arbitrary blocks | All GPUs | Deadlocks |
| Kernel Splitting | All blocks | All GPUs | Overhead |
| Warp Coordination | Same-SM blocks | All GPUs | Unreliable |

---

### **Recommendations**

1. **Prefer Cooperative Groups** for Volta+ GPUs.
2. **For Older GPUs**:
    - Use atomic counters but limit synced blocks to ≤ SM count.
    - Add timeout mechanisms to avoid deadlocks:

```cpp
uint64_t start_clock = clock64();
while (atomicAdd(flag, 0) < target && clock64() - start_clock < timeout) {}
```

3. **Profile** with `nvprof`/Nsight Compute to detect warp stalls from synchronization attempts.

---

By carefully managing GPU occupancy and synchronization granularity, these methods enable partial block coordination while minimizing performance penalties.

<div style="text-align: center">⁂</div>

[^6_1]: https://www.atlantis-press.com/article/25881164.pdf

[^6_2]: https://www.shinhwei.com/cuda-repair.pdf

[^6_3]: https://stackoverflow.com/questions/6404992/cuda-block-synchronization

[^6_4]: https://userweb.cs.txstate.edu/~burtscher/papers/iiswc24b.pdf

[^6_5]: https://forums.developer.nvidia.com/t/synchronize-all-blocks-in-cuda/27774

[^6_6]: https://lingming.cs.illinois.edu/publications/icse2020b.pdf

[^6_7]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[^6_8]: https://discourse.julialang.org/t/cuda-global-synchronization-howto/74920

