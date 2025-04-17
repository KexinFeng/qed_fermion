import torch
from typing import Dict, List
from vllm.worker.model_runner import CUDAGraphRunner, _get_graph_batch_size


class GraphRunner(CUDAGraphRunner):
    def __init__(self, function):
        self.function = function
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        probs: torch.Tensor,
        draft_tokens: torch.Tensor,        
        tree,
        first_successor_idx,
        prefix_length,
        block_tables,
        block_size,
        draft_kv_caches: List[torch.Tensor],
        target_kv_caches: List[torch.Tensor],
        graph_memory_pool,
        n_warmups = 1
    ) -> None:
        input_buffers={
            "probs": probs,
            "draft_tokens": draft_tokens,
            "prefix_length": prefix_length,
            "block_tables": block_tables
            }

        # Warm up
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                static_outputs = self.function(
                    input_buffers['probs'], 
                    input_buffers['draft_tokens'],
                    tree.parent_group_idx, 
                    tree.bin_filter, 
                    tree.parent_idx, 
                    tree.size,
                    tree.depth,
                    tree.successor_tail_idx,
                    tree.successor_tail_idx_stablized,
                    first_successor_idx,
                    probs.size(0),
                    input_buffers['prefix_length'],
                    input_buffers['block_tables'],
                    block_size, 
                    draft_kv_caches, 
                    target_kv_caches
                )
            s.synchronize()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.function(
                    input_buffers['probs'], 
                    input_buffers['draft_tokens'],
                    tree.parent_group_idx, 
                    tree.bin_filter, 
                    tree.parent_idx, 
                    tree.size,
                    tree.depth,
                    tree.successor_tail_idx,
                    tree.successor_tail_idx_stablized,
                    first_successor_idx,
                    probs.size(0),
                    input_buffers['prefix_length'],
                    input_buffers['block_tables'],
                    block_size, 
                    draft_kv_caches, 
                    target_kv_caches
                )
        self.graph = graph
        self.input_buffers = input_buffers
        self.output_buffers = {"outputs": static_outputs}
        return graph_memory_pool
        
    def __call__(
        self,
        probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        prefix_length: torch.Tensor,
        block_tables: torch.Tensor,
        draft_kv_caches,
        target_kv_caches
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del draft_kv_caches, target_kv_caches

        self.input_buffers['probs'].copy_(probs)
        self.input_buffers['draft_tokens'].copy_(draft_tokens)
        self.input_buffers['prefix_length'].copy_(prefix_length)
        self.input_buffers['block_tables'].copy_(block_tables)

        self.graph.replay()
        # output_tokens, output_token_idx, acc_lengths, probs_eval, seq_id_bonus
        return self.output_buffers['outputs']


class DraftGraphRunner2D(CUDAGraphRunner):
    def __init__(self, runner):
        self.runner = runner
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        draft_tokens,
        positions,
        kv_caches,
        attn_metadata,
        tree,
        batch_size,
        grow_step,
        cu_idx,
        layer_size,
        graph_memory_pool,
        n_warmups = 1
    ) -> None:
        # Here attn_metadata.slot_mapping and attn_metadata.tree_mask will be reasigned inside drafting(), so it's important to store its input in input_buffers. 
        # But it doesn't mean that tree_mask entries need updating in a graph call. Similar to kv_cache, tree_mask is also static.

        # Save the input and output buffers.
        self.input_buffers = {
            "draft_tokens": draft_tokens,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": attn_metadata.slot_mapping,
            "tree_mask": attn_metadata.tree_mask,
            "context_lens": attn_metadata.context_lens,
            "block_tables": attn_metadata.block_tables,
        }

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmups):
                static_outputs = self.runner.drafting(
                    draft_tokens, 
                    self.input_buffers['slot_mapping'],
                    positions,
                    self.input_buffers['tree_mask'],
                    kv_caches, 
                    attn_metadata, 
                    tree, batch_size, grow_step, cu_idx, layer_size      
                )
            s.synchronize()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_memory_pool):
            static_outputs = self.runner.drafting(
                draft_tokens, 
                self.input_buffers['slot_mapping'],
                positions,
                self.input_buffers['tree_mask'],
                kv_caches, 
                attn_metadata,  
                tree, batch_size, grow_step, cu_idx, layer_size  
            )
        self.graph = graph
        self.output_buffers = {"outputs": static_outputs}
        return graph_memory_pool

    def __call__(
        self,
        draft_tokens: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        tree_mask: torch.Tensor,
        attn_metadata
    ) -> torch.Tensor:
        self.input_buffers["draft_tokens"].copy_(draft_tokens, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["context_lens"].copy_(attn_metadata.context_lens,
                                                 non_blocking=True)
        self.input_buffers["block_tables"].copy_(attn_metadata.block_tables,
                                                 non_blocking=True)
        self.input_buffers["tree_mask"].copy_(tree_mask, 
                                                non_blocking=True)
        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["outputs"]

