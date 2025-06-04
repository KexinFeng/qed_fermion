import torch
import os
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.utils.util import device_mem

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
        max_iter_se,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the get_fermion_obsr function execution as a CUDA graph."""
        input_buffers = {
            "bosons": bosons,
            "eta": eta
        }

        max_iter_copy = self.se.hmc_sampler.max_iter
        self.se.hmc_sampler.max_iter = max_iter_se
        
        # Warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

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
        start_mem = device_mem()[1]
        print(f"Inside capture start_mem: {start_mem:.2f} MB\n")

        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.se.get_fermion_obsr(
                input_buffers['bosons'],
                input_buffers['eta']
            )

        end_mem = device_mem()[1]   
        print(f"Inside capture end_mem: {end_mem:.2f} MB\n")
        print(f"fermion_obsr CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")
    
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
