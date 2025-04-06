## Sync

`cudaStreamSynchronize` is a CUDA runtime API function that blocks the host (CPU) until all tasks in a specified CUDA stream have completed. CUDA streams are used to manage and execute tasks (like kernel launches, memory transfers, etc.) asynchronously on the GPU. By default, operations in a stream are executed in order, but they do not block the host or other streams.

### Syntax:
```cpp
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
```

### Parameters:
- **`stream`**: The CUDA stream to synchronize. If you pass `0` (or `cudaStreamDefault`), it synchronizes the default stream.

### Behavior:
- The function ensures that all previously issued commands in the specified stream are completed before the host thread continues execution.
- If there are no pending tasks in the stream, the function returns immediately.

### Example:
```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel() {
    // Simple kernel doing some work
    printf("Hello from GPU!\n");
}

int main() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch kernel in the created stream
    kernel<<<1, 1, 0, stream>>>();

    // Synchronize the stream to ensure kernel execution is complete
    cudaStreamSynchronize(stream);

    std::cout << "Kernel execution completed!" << std::endl;

    cudaStreamDestroy(stream);
    return 0;
}
```

### Key Points:
1. **Blocking Behavior**: The host thread will wait until all tasks in the stream are finished. This is useful when you need to ensure that GPU computations are complete before proceeding with host-side logic.
2. **Performance Consideration**: Overusing `cudaStreamSynchronize` can reduce performance because it forces the host to wait. Use it only when necessary.
3. **Error Handling**: Always check the return value of `cudaStreamSynchronize` to handle potential errors like invalid streams or device-side failures.

### Common Use Case:
- Synchronizing streams before accessing GPU results on the host.
- Debugging to ensure tasks are completed in a specific order.

Let me know if you'd like further clarification!