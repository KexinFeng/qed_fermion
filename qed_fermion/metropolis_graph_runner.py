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
        self.input_buffers = None
        self.output_buffers = None
    
    def capture(
        self,
        boson,
        tau_mask,
        max_iter,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the leapfrog_proposer5_cmptau_graphrun function execution as a CUDA graph."""

        input_buffers = {
            "boson": boson,
            "tau_mask": tau_mask
        }

        self.hmc_sampler.max_iter = max_iter
        
        # Warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                static_outputs = self.hmc_sampler.leapfrog_proposer5_cmptau_graphrun(
                    input_buffers['boson'],
                    input_buffers['tau_mask']
                )

            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)
        start_mem = device_mem()[1]
        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.hmc_sampler.leapfrog_proposer5_cmptau_graphrun(
                input_buffers['boson'],
                input_buffers['tau_mask']
            )

        end_mem = device_mem()[1]   
        print(f"leapfrog_proposer5_cmptau_graphrun CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")
    
        self.graph = graph
        # No input buffers needed for this method - it uses internal state
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs  # (boson_new, H_old, H_new, cg_converge_iter, cg_r_err)
                
        return graph.pool()
    
    def __call__(self, boson, tau_mask):
        """Execute the captured graph."""
        self.input_buffers['boson'].copy_(boson)
        self.input_buffers['tau_mask'].copy_(tau_mask)

        # Replay the graph
        self.graph.replay()

        # Return the outputs: (boson_new, H_old, H_new, cg_converge_iter, cg_r_err)
        # Return the outputs
        return self.output_buffers
    
class LeapfrogCmpGraphRunner:
    def __init__(self, hmc_sampler):
        self.hmc_sampler = hmc_sampler
        self.graph = None
        self.input_buffers = None
        self.output_buffers = None

    def capture(
        self,
        x,
        p,
        dt,
        tau_mask,
        force_b_plaq,
        force_b_tau,
        max_iter,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the leapfrog_cmp_graph function execution as a CUDA graph."""

        input_buffers = {
            "x": x,
            "p": p,
            "dt": dt,
            "tau_mask": tau_mask,
            "force_b_plaq": force_b_plaq,
            "force_b_tau": force_b_tau
        }

        self.hmc_sampler.max_iter = max_iter

        # Warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                static_outputs = self.hmc_sampler.leapfrog_cmp_graph(
                    input_buffers['x'],
                    input_buffers['p'],
                    input_buffers['dt'],
                    input_buffers['tau_mask'],
                    input_buffers['force_b_plaq'],
                    input_buffers['force_b_tau']
                )
            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)
        start_mem = device_mem()[1]
        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.hmc_sampler.leapfrog_cmp_graph(
                input_buffers['x'],
                input_buffers['p'],
                input_buffers['dt'],
                input_buffers['tau_mask'],
                input_buffers['force_b_plaq'],
                input_buffers['force_b_tau']
            )

        end_mem = device_mem()[1]
        print(f"leapfrog_cmp_graph CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")

        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs

        return graph.pool()

    def __call__(self, x, p, dt, tau_mask, force_b_plaq, force_b_tau):
        """Execute the captured graph."""
        self.input_buffers['x'].copy_(x)
        self.input_buffers['p'].copy_(p)
        self.input_buffers['dt'].copy_(dt)
        self.input_buffers['tau_mask'].copy_(tau_mask)
        self.input_buffers['force_b_plaq'].copy_(force_b_plaq)
        self.input_buffers['force_b_tau'].copy_(force_b_tau)

        # Replay the graph
        self.graph.replay()

        # Return the outputs
        return self.output_buffers

