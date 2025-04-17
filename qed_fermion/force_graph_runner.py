import torch

class ForceGraphRunner:
    def __init__(self, hmc_sampler):
        self.hmc_sampler = hmc_sampler
        self.graph = None
        self.input_buffers = {}
        self.output_buffers = {}
    
    def capture(
        self,
        psi,
        boson,
        graph_memory_pool=None,
        n_warmups=1
    ):
        """Capture the force_f_fast function execution as a CUDA graph."""
        input_buffers = {
            "psi": psi,
            "boson": boson
        }
        
        # Warm up
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                static_outputs = self.hmc_sampler.force_f_fast(
                    input_buffers['psi'],
                    input_buffers['boson'],
                    None  # MhM is not used in the actual computation
                )
            s.synchronize()
        torch.cuda.current_stream().wait_stream(s)
        
        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.hmc_sampler.force_f_fast(
                input_buffers['psi'],
                input_buffers['boson'],
                None  # MhM is not used in the actual computation
            )
        
        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = {
            "Ft": static_outputs[0],
            "xi_t": static_outputs[1],
            "cg_converge_iter": static_outputs[2]
        }
        
        return graph.pool()
    
    def __call__(
        self,
        psi,
        boson
    ):
        """Execute the captured graph with the given inputs."""
        # Copy inputs to input buffers
        self.input_buffers['psi'].copy_(psi)
        self.input_buffers['boson'].copy_(boson)
        
        # Replay the graph
        self.graph.replay()
        
        # Return the outputs
        return (
            self.output_buffers['Ft'],
            self.output_buffers['xi_t'],
            self.output_buffers['cg_converge_iter']
        )