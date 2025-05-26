import torch

class FermionObsrGraphRunner:
    def __init__(self, se):
        self.graph = None
        self.se = se
        self.input_buffers = {}
        self.output_buffers = {}
    
    def capture(
        self,
        bosons,
        eta,
        # cuda_graph control parameters
        max_iter,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the get_fermion_obsr function execution as a CUDA graph."""
        input_buffers = {
            "bosons": bosons,
            "eta": eta
        }

        self.se.hmc_sampler.max_iter, max_iter_copy = max_iter, self.se.hmc_sampler.max_iter
        
        # Warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                static_outputs = self.se.get_fermion_obsr(
                    input_buffers['bosons'],
                    input_buffers['eta']
                )

            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)
        
        # Memory
        start_mem = torch.cuda.memory_allocated()

        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.se.get_fermion_obsr(
                input_buffers['bosons'],
                input_buffers['eta']
            )

        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        graph_footage = end_mem - start_mem
        print(f"get_fermion_obsr CUDA Graph memory footage: {graph_footage / 1024**2:.2f} MB")
        print(f"get_fermion_obsr CUDA Graph peak memory: {peak_mem / 1024**2:.2f} MB")
    
        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs

        self.se.hmc_sampler.max_iter = max_iter_copy
        return graph.pool()
    
    def __call__(
        self,
        bosons,
        eta,
    ):
        """Execute the captured graph with the given inputs."""
        # Copy inputs to input buffers
        self.input_buffers['bosons'].copy_(bosons)
        self.input_buffers['eta'].copy_(eta)

        # Replay the graph
        self.graph.replay()
        
        # Return the outputs
        return self.output_buffers
