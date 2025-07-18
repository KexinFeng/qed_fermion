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
        bosons, # [bs, 2, Lx, Ly, Ltau] 
        eta,    # [Nrv, Ltau * Ly * Lx]
        indices, # [C(Nrv, 4), 4]
        indices_r2, # [C(Nrv, 2), 2]
        # cuda_graph control parameters
        max_iter_se,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the get_fermion_obsr function execution as a CUDA graph."""
        input_buffers = {
            "bosons": bosons,
            "eta": eta, 
            "indices": indices,
            "indices_r2": indices_r2
        }

        max_iter_copy = self.se.hmc_sampler.max_iter
        self.se.hmc_sampler.max_iter = max_iter_se
        
        # Warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Memory
        start_mem = device_mem()[1]
        print(f"Inside capture start_mem_0: {start_mem:.2f} MB\n")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for n in range(n_warmups):
                static_outputs = self.se.get_fermion_obsr(
                    input_buffers['bosons'],
                    input_buffers['eta'],
                    input_buffers['indices'],
                    input_buffers['indices_r2']
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
                input_buffers['eta'],
                input_buffers['indices'],
                input_buffers['indices_r2']
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
        bosons, # [bs, 2, Lx, Ly, Ltau] 
        eta,    # [Nrv, Ltau * Ly * Lx]
        indices,  # [C(Nrv, 4), 4] (indices for the stochastic estimator, e.g., sa, sb, sc, sd)
        indices_r2  # [C(Nrv, 2), 2] (indices for the stochastic estimator, e.g., sa, sb)
    ):
        """Execute the captured graph with the given inputs."""
        # Copy inputs to input buffers
        self.input_buffers['bosons'].copy_(bosons)
        self.input_buffers['eta'].copy_(eta)
        if indices is not None:
            self.input_buffers['indices'].copy_(indices)
        if indices_r2 is not None:
            self.input_buffers['indices_r2'].copy_(indices_r2)

        # Replay the graph
        self.graph.replay()
        
        # Return the outputs
        return self.output_buffers


class GEtaGraphRunner:
    def __init__(self, se):
        self.graph = None
        self.se = se
        self.input_buffers = {}
        self.output_buffers = {}

    def capture(
        self,
        max_iter_se,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the get_fermion_obsr function execution as a CUDA graph."""
        input_buffers = {
            "boson": torch.zeros((1, 2, self.se.Lx, self.se.Ly, self.se.Ltau), device=self.se.device, dtype=self.se.dtype),
            "eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Vs), device=self.se.device, dtype=self.se.cdtype),
        }

        max_iter_copy = self.se.hmc_sampler.max_iter
        self.se.hmc_sampler.max_iter = max_iter_se

        # Warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Memory
        start_mem = device_mem()[1]
        print(f"Inside capture start_mem_0: {start_mem:.2f} MB\n")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for n in range(n_warmups):
                static_outputs = self.se.set_eta_G_eta_inner(
                    input_buffers['boson'],
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
            static_outputs = self.se.set_eta_G_eta_inner(
                input_buffers['boson'],
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
        boson, # [bs, 2, Lx, Ly, Ltau]
        eta,    # [Nrv, Ltau * Ly * Lx]
    ):
        """Execute the captured graph with the given inputs."""
        # Copy inputs to input buffers
        self.input_buffers['boson'].copy_(boson)
        self.input_buffers['eta'].copy_(eta)

        # Replay the graph
        self.graph.replay()

        # Return the outputs
        return self.output_buffers


class L0GraphRunner:
    def __init__(self, se):
        self.graph = None
        self.se = se
        self.input_buffers = {}
        self.output_buffers = {}

    def capture(
        self,
        params,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the get_fermion_obsr function execution as a CUDA graph."""
        input_buffers = {
            "G_eta": torch.zeros((self.se.Nrv, self.se.Ltau, self.se.Ly, self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "eta_conj": torch.zeros((self.se.Nrv, self.se.Ltau, self.se.Ly, self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "sa_chunk": torch.zeros((params.outer_stride,), device=self.se.device, dtype=torch.int64),
            "sb_chunk": torch.zeros((params.outer_stride,), device=self.se.device, dtype=torch.int64),
            "sc_chunk": torch.zeros((params.outer_stride,), device=self.se.device, dtype=torch.int64),
            "sd_chunk": torch.zeros((params.outer_stride,), device=self.se.device, dtype=torch.int64)
        }

        # Warm up
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Memory
        start_mem = device_mem()[1]
        print(f"Inside capture start_mem_0: {start_mem:.2f} MB\n")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for n in range(n_warmups):
                static_outputs = self.se.L0_inner(
                    input_buffers['sa_chunk'],
                    input_buffers['sb_chunk'],
                    input_buffers['sc_chunk'],
                    input_buffers['sd_chunk'],
                    input_buffers['G_eta'],
                    input_buffers['eta_conj'],
                    params
                )

            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)

        # Memory
        start_mem = device_mem()[1]
        print(f"Inside capture start_mem: {start_mem:.2f} MB\n")

        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.se.L0_inner(
                input_buffers['sa_chunk'],
                input_buffers['sb_chunk'],
                input_buffers['sc_chunk'],
                input_buffers['sd_chunk'],
                input_buffers['G_eta'],
                input_buffers['eta_conj'],
                params
            )

        end_mem = device_mem()[1]
        print(f"Inside capture end_mem: {end_mem:.2f} MB\n")
        print(f"fermion_obsr CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")

        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs

        return graph.pool()

    def __call__(
        self,
        sa_chunk,
        sb_chunk,
        sc_chunk,
        sd_chunk,
        G_eta,       # [bs=1, 2, Lx, Ly, Ltau]
        eta_conj,    # [Nrv, Ltau * Ly * Lx]
    ):
        """Execute the captured graph with the given inputs."""
        # Copy inputs to input buffers
        self.input_buffers['G_eta'].copy_(G_eta)
        self.input_buffers['eta_conj'].copy_(eta_conj)
        self.input_buffers['sa_chunk'].copy_(sa_chunk)
        self.input_buffers['sb_chunk'].copy_(sb_chunk)
        self.input_buffers['sc_chunk'].copy_(sc_chunk)
        self.input_buffers['sd_chunk'].copy_(sd_chunk)

        # Replay the graph
        self.graph.replay()

        # Return the outputs
        return self.output_buffers


class SpsmGraphRunner:
    def __init__(self, se):
        self.graph = None
        self.se = se
        self.input_buffers = {}
        self.output_buffers = None

    def capture(
        self,
        graph_memory_pool=None,
        n_warmups=3
    ):
        """Capture the spsm_r_util function execution as a CUDA graph."""
        input_buffers = {
            "eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "G_eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_mem = device_mem()[1]
        print(f"Inside SpsmGraphRunner capture start_mem_0: {start_mem:.2f} MB\n")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for n in range(n_warmups):
                static_outputs = self.se.spsm_r_util(
                    input_buffers['eta'],
                    input_buffers['G_eta']
                )
            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)

        start_mem = device_mem()[1]
        print(f"Inside SpsmGraphRunner capture start_mem: {start_mem:.2f} MB\n")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.se.spsm_r_util(
                input_buffers['eta'],
                input_buffers['G_eta']
            )

        end_mem = device_mem()[1]
        print(f"Inside SpsmGraphRunner capture end_mem: {end_mem:.2f} MB\n")
        print(f"SpsmGraphRunner CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")

        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs

        return graph.pool()

    def __call__(
        self,
        eta,    # [Nrv, Ltau * Ly * Lx]
        G_eta   # [Nrv, Ltau * Ly * Lx]
    ):
        """Execute the captured graph with the given inputs."""
        self.input_buffers['G_eta'].copy_(G_eta)
        self.input_buffers['eta'].copy_(eta)

        self.graph.replay()
        return self.output_buffers

class T1GraphRunner:
    def __init__(self, se):
        self.graph = None
        self.se = se
        self.input_buffers = {}
        self.output_buffers = None

    def capture(
        self,
        graph_memory_pool=None,
        n_warmups=3
    ):
        input_buffers = {
            "eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "G_eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "boson": torch.zeros((1, 2, self.se.Ltau, self.se.Ly, self.se.Lx), device=self.se.device, dtype=self.se.dtype),
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_mem = device_mem()[1]
        print(f"Inside T1GraphRunner capture start_mem_0: {start_mem:.2f} MB\n")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for n in range(n_warmups):
                static_outputs = self.se.T1(
                    input_buffers['eta'],
                    input_buffers['G_eta'],
                    input_buffers['boson']
                )
            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)

        start_mem = device_mem()[1]
        print(f"Inside T1GraphRunner capture start_mem: {start_mem:.2f} MB\n")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.se.T1(
                input_buffers['eta'],
                input_buffers['G_eta'],
                input_buffers['boson']
            )

        end_mem = device_mem()[1]
        print(f"Inside T1GraphRunner capture end_mem: {end_mem:.2f} MB\n")
        print(f"T1GraphRunner CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")

        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs

        return graph.pool()

    def __call__(
        self,
        eta,    # [Nrv, Ltau * Ly * Lx]
        G_eta,  # [Nrv, Ltau * Ly * Lx]
        boson   # [1, 2, Ltau, Ly, Lx]
    ):
        self.input_buffers['eta'].copy_(eta)
        self.input_buffers['G_eta'].copy_(G_eta)
        self.input_buffers['boson'].copy_(boson)
        self.graph.replay()
        return self.output_buffers

class T2GraphRunner:
    def __init__(self, se):
        self.graph = None
        self.se = se
        self.input_buffers = {}
        self.output_buffers = None

    def capture(
        self,
        graph_memory_pool=None,
        n_warmups=3
    ):
        input_buffers = {
            "eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "G_eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "boson": torch.zeros((1, 2, self.se.Ltau, self.se.Ly, self.se.Lx), device=self.se.device, dtype=self.se.dtype),
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_mem = device_mem()[1]
        print(f"Inside T2GraphRunner capture start_mem_0: {start_mem:.2f} MB\n")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for n in range(n_warmups):
                static_outputs = self.se.T2(
                    input_buffers['eta'],
                    input_buffers['G_eta'],
                    input_buffers['boson']
                )
            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)

        start_mem = device_mem()[1]
        print(f"Inside T2GraphRunner capture start_mem: {start_mem:.2f} MB\n")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.se.T2(
                input_buffers['eta'],
                input_buffers['G_eta'],
                input_buffers['boson']
            )

        end_mem = device_mem()[1]
        print(f"Inside T2GraphRunner capture end_mem: {end_mem:.2f} MB\n")
        print(f"T2GraphRunner CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")

        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs

        return graph.pool()

    def __call__(
        self,
        eta,    # [Nrv, Ltau * Ly * Lx]
        G_eta,  # [Nrv, Ltau * Ly * Lx]
        boson   # [1, 2, Ltau, Ly, Lx]
    ):
        self.input_buffers['eta'].copy_(eta)
        self.input_buffers['G_eta'].copy_(G_eta)
        self.input_buffers['boson'].copy_(boson)
        self.graph.replay()
        return self.output_buffers

class T3GraphRunner:
    def __init__(self, se):
        self.graph = None
        self.se = se
        self.input_buffers = {}
        self.output_buffers = None

    def capture(
        self,
        graph_memory_pool=None,
        n_warmups=3
    ):
        input_buffers = {
            "eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "G_eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "boson": torch.zeros((1, 2, self.se.Ltau, self.se.Ly, self.se.Lx), device=self.se.device, dtype=self.se.dtype),
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_mem = device_mem()[1]
        print(f"Inside T3GraphRunner capture start_mem_0: {start_mem:.2f} MB\n")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for n in range(n_warmups):
                static_outputs = self.se.T3(
                    input_buffers['eta'],
                    input_buffers['G_eta'],
                    input_buffers['boson']
                )
            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)

        start_mem = device_mem()[1]
        print(f"Inside T3GraphRunner capture start_mem: {start_mem:.2f} MB\n")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.se.T3(
                input_buffers['eta'],
                input_buffers['G_eta'],
                input_buffers['boson']
            )

        end_mem = device_mem()[1]
        print(f"Inside T3GraphRunner capture end_mem: {end_mem:.2f} MB\n")
        print(f"T3GraphRunner CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")

        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs

        return graph.pool()

    def __call__(
        self,
        eta,    # [Nrv, Ltau * Ly * Lx]
        G_eta,  # [Nrv, Ltau * Ly * Lx]
        boson   # [1, 2, Ltau, Ly, Lx]
    ):
        self.input_buffers['eta'].copy_(eta)
        self.input_buffers['G_eta'].copy_(G_eta)
        self.input_buffers['boson'].copy_(boson)
        self.graph.replay()
        return self.output_buffers

class T4GraphRunner:
    def __init__(self, se):
        self.graph = None
        self.se = se
        self.input_buffers = {}
        self.output_buffers = None

    def capture(
        self,
        graph_memory_pool=None,
        n_warmups=3
    ):
        input_buffers = {
            "eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "G_eta": torch.zeros((self.se.Nrv, self.se.Ltau * self.se.Ly * self.se.Lx), device=self.se.device, dtype=self.se.cdtype),
            "boson": torch.zeros((1, 2, self.se.Ltau, self.se.Ly, self.se.Lx), device=self.se.device, dtype=self.se.dtype),
        }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start_mem = device_mem()[1]
        print(f"Inside T4GraphRunner capture start_mem_0: {start_mem:.2f} MB\n")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for n in range(n_warmups):
                static_outputs = self.se.T4(
                    input_buffers['eta'],
                    input_buffers['G_eta'],
                    input_buffers['boson']
                )
            s.synchronize()

        torch.cuda.current_stream().wait_stream(s)

        start_mem = device_mem()[1]
        print(f"Inside T4GraphRunner capture start_mem: {start_mem:.2f} MB\n")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.se.T4(
                input_buffers['eta'],
                input_buffers['G_eta'],
                input_buffers['boson']
            )

        end_mem = device_mem()[1]
        print(f"Inside T4GraphRunner capture end_mem: {end_mem:.2f} MB\n")
        print(f"T4GraphRunner CUDA Graph diff: {end_mem - start_mem:.2f} MB\n")

        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = static_outputs

        return graph.pool()

    def __call__(
        self,
        eta,    # [Nrv, Ltau * Ly * Lx]
        G_eta,  # [Nrv, Ltau * Ly * Lx]
        boson   # [1, 2, Ltau, Ly, Lx]
    ):
        self.input_buffers['eta'].copy_(eta)
        self.input_buffers['G_eta'].copy_(G_eta)
        self.input_buffers['boson'].copy_(boson)
        self.graph.replay()
        return self.output_buffers




