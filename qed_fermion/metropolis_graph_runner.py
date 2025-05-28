import torch
import os
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path + '/../')

from qed_fermion.utils.util import device_mem

class MetropolisGraphRunner:
    def __init__(self, hmc_sampler):
        self.hmc_sampler = hmc_sampler
        self.graph = None
        self.input_buffers = {}
        self.output_buffers = {}
    
    def capture(
        self,
        max_iter,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the leapfrog_proposer5_cmptau_graphrun function execution as a CUDA graph."""
        
        # Save the original max_iter and set the new one
        original_max_iter = self.hmc_sampler.max_iter
        self.hmc_sampler.max_iter = max_iter
        
        # Warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                static_outputs = self.hmc_sampler.leapfrog_proposer5_cmptau_graphrun()

            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)

        start_mem = device_mem()[1]
        
        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.hmc_sampler.leapfrog_proposer5_cmptau_graphrun()

        end_mem = device_mem()[1]   
        print(f"leapfrog_proposer5_cmptau_graphrun CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")
    
        self.graph = graph
        # No input buffers needed for this method - it uses internal state
        self.input_buffers = {}
        self.output_buffers = static_outputs  # (boson_new, H_old, H_new, cg_converge_iter, cg_r_err)
        
        # Restore the original max_iter
        self.hmc_sampler.max_iter = original_max_iter
        
        return graph.pool()
    
    def __call__(self):
        """Execute the captured graph."""
        # Replay the graph
        self.graph.replay()
        
        # Return the outputs: (boson_new, H_old, H_new, cg_converge_iter, cg_r_err)
        return self.output_buffers
