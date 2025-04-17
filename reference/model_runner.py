import vllm.sequence
from lmi_dist.patch.speculative.graph_runner import DraftGraphRunner2D, GraphRunner
from lmi_dist.patch.speculative.tree import Tree, cu_add
import lmi_dist.worker.model_runner
import numpy as np
import torch
import time

from typing import Dict, List, Tuple, Optional, Set
import vllm.model_executor.parallel_utils.parallel_state

from .sequence import SpeculativeSpeculateOutput, SpeculativeSpeculateSequenceGroupOutput
from .sampling_metadata import SpeculativeSamplingMetadata
from .sampler import SpeculativeSampler, SpeculativeSamplerWithLoRA, SpeculativeLogitsProcessor
from ...utils.replace_module import replace_module
from .parallel_state import ActiveModel, MarkActiveModel

from vllm.worker.model_runner import CUDAGraphRunner, _get_graph_batch_size
from vllm.utils import make_tensor_with_pad, CudaMemoryProfiler
from vllm.model_executor.model_loader import get_model
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping, LogitsProcessorWithLoRA
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.parallel_utils import pynccl_utils
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.parallel_utils.communication_op import tensor_model_parallel_all_gather
import vllm.model_executor.layers.sampler as vllm_sampler
from lmi_dist._C import rubikon_cache_ops

from vllm.sequence import SequenceGroupMetadata, SequenceData, Logprob
import torch.distributed as dist

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4, 6] + [8 * i for i in range(1, 33)]
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4, 6] + [8 * i for i in range(1, 9)]
# _BATCH_SIZES_TO_CAPTURE = [1, 2, 4, 8, 16, 24, 32]
# _BATCH_SIZES_TO_CAPTURE = [1, 2, 4, 8, 16, 32]
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4, 8, 16]  # cap_size = 16, batch_size up to 2
# _BATCH_SIZES_TO_CAPTURE = [1, 2]   # batch_size = 16, cap_size up to 2
# _BATCH_SIZES_TO_CAPTURE = [1, 2, 4]   # cap_size = 4, batch_size up to 2
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4, 8, 12, 16]  # cap_size = 16, batch_size up to 2
LORA_WARMUP_RANK = 8

LOG_0 = -1000
DEBUG = False
_debug = False

def get_speculative_sampler(sampler: Sampler):
    return SpeculativeSampler()


def get_replacement_logits_processor(logits_processor: LogitsProcessor):
    return SpeculativeLogitsProcessor(logits_processor.vocab_size,
                                      logits_processor.org_vocab_size,
                                      logits_processor.scale,
                                      logits_processor.logits_as_input)


def get_replacement_sampler_with_lora(sampler: LogitsProcessorWithLoRA):
    sampler.base_layer = SpeculativeSamplerWithLoRA()
    return sampler


class SpeculativeModelRunner(lmi_dist.worker.model_runner.VllmModelRunner):
    def __init__(self, *args, **kwargs):
        # Variables used in speculative decoding
        self.rank = kwargs.pop('rank', 0)
        self.draft_model = None
        self.draft_model_config = kwargs.pop('draft_model_config', None)
        self.speculate_length = kwargs.pop('speculate_length', 5)
        self.speculate_tree = kwargs.pop('speculate_tree', None)
        self.num_multi_query_mode_tokens = self.speculate_length + 1
        self.draft_graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.draft_graph_memory_pool = None  # Set during graph capture
        self.tree_verification_graph_runners = {}
        self.tree_verification_executors_memory_pool = None

        super().__init__(*args, **kwargs)

        # Update module replacements to replace sampler
        # TODO Add LoRA support with speculative decoding.
        self.module_replacements[Sampler] = get_speculative_sampler
        self.module_replacements[
            LogitsProcessor] = get_replacement_logits_processor

    def load_model(self) -> None:
        super().load_model()
        with MarkActiveModel(ActiveModel.DRAFT):
            # NOTE: We use global variable to control parallel state (world size, rank)
            # of draft model used in speculative decoding.
            with CudaMemoryProfiler() as m:
                self.draft_model = get_model(self.draft_model_config,
                                             self.device_config)
            self.draft_model_memory_usage = m.consumed_memory
            logger.info(
                f"Loading draft model weights took "
                f"{self.draft_model_memory_usage / float(2**30):.4f} GB")

    def get_max_block_per_batch(self) -> int:
        # TODO: Is this correct?
        num_padded_tokens = self.num_multi_query_mode_tokens
        max_num_blocks = (self.max_context_len_to_capture + num_padded_tokens +
                          self.block_size - 1) // self.block_size
        return max_num_blocks

    @torch.inference_mode()
    def speculate_capture_model(self, kv_caches: List[torch.Tensor],
                                draft_kv_caches: List[torch.Tensor]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        # NOTE(woosuk): This is a hack to ensure that the NCCL backend is never
        # deleted before the CUDA graphs.
        self.pynccl_backend = pynccl_utils.get_nccl_backend()

        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        def capture_graph_inner(is_draft_model: bool = False,
                                caches: List[torch.Tensor] = None) -> None:
            tree_mask = None
            if is_draft_model:
                seq_len = 1
                is_multi_query_mode = False
                active_model = ActiveModel.DRAFT
                graph_runners = self.draft_graph_runners
                graph_memory_pool = self.draft_graph_memory_pool
            else:
                seq_len = self.speculate_length + 1  # TODO: change to tree.size
                if self.speculate_tree:
                    last_layer_size = len(self.speculate_tree.grow_map['roots'][-1])
                    input_size = self.speculate_tree.size - last_layer_size
                    seq_len = input_size
                    tree_mask = torch.ones(seq_len,
                                       seq_len,
                                       dtype=torch.bool,
                                       device="cuda")
                    
                is_multi_query_mode = True
                active_model = ActiveModel.TARGET
                graph_runners = self.graph_runners
                graph_memory_pool = self.graph_memory_pool

            # Prepare dummy inputs. These will be reused for all batch sizes.
            input_tokens = torch.zeros(max_batch_size,
                                       seq_len,
                                       dtype=torch.long,
                                       device="cuda")
            input_positions = torch.zeros(max_batch_size,
                                          seq_len,
                                          dtype=torch.long,
                                          device="cuda")
            slot_mapping = torch.empty(max_batch_size,
                                       seq_len,
                                       dtype=torch.long,
                                       device="cuda")
            slot_mapping.fill_(_PAD_SLOT_ID)
            context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
            block_tables = torch.from_numpy(self.graph_block_tables).cuda()

            # NOTE(woosuk): There are 3 backends for all-reduce: custom all-reduce
            # kernel, CuPy NCCL, and PyTorch NCCL. When using CUDA graph, we use
            # either custom all-reduce kernel or CuPy NCCL. When not using CUDA
            # graph, we use either custom all-reduce kernel or PyTorch NCCL.
            # We always prioritize using custom all-reduce kernel but fall back
            # to PyTorch or CuPy NCCL if it is disabled or not supported.
            with custom_all_reduce.capture():
                # NOTE: Capturing the largest batch size first may help reduce the
                # memory usage of CUDA graph.
                for batch_size in reversed(batch_size_capture_list):
                    # Create dummy attn_metadata.
                    attn_metadata = self.attn_backend.make_metadata(
                        is_prompt=False,
                        # NOTE: slot_mapping is 2d, we need to flatten it to
                        # 1d after slicing.
                        slot_mapping=slot_mapping[:batch_size].view(-1),
                        prompt_lens=None,
                        prompt_lens_tensor=None,
                        num_prompt_tokens=0,
                        num_generation_tokens=0,
                        max_subquery_len=None,
                        max_context_len=self.max_context_len_to_capture,
                        max_prompt_len=None,
                        subquery_start_loc=None,
                        seq_start_loc=None,
                        context_lens=context_lens[:batch_size],
                        block_tables=block_tables[:batch_size],
                        use_cuda_graph=True,
                        kv_cache_dtype=self.kv_cache_dtype,
                        seq_len=seq_len,
                        is_multi_query_mode=is_multi_query_mode,
                        tree_mask=tree_mask
                    )
                    # assert tree_mask is None

                    # TODO: the input_tokens and input_positions here a local. caches are global
                    # tree_attn will be global. attn_metadata contains context lengths but a finer cu_siblings is needed.
                    graph_runner = CUDAGraphRunner(self.draft_model) if is_draft_model \
                        else CUDAGraphRunner(self.model)
                    with MarkActiveModel(active_model):
                        graph_runner.capture(
                            # NOTE: input_tokens and input_positions are 2d, we need to
                            # flatten them to 1d after slicing.
                            input_tokens[:batch_size].view(-1),
                            input_positions[:batch_size].view(-1),
                            caches,
                            attn_metadata,
                            memory_pool=graph_memory_pool,
                        )
                    graph_memory_pool = graph_runner.graph.pool()
                    graph_runners[batch_size] = graph_runner
                if is_draft_model:
                    self.draft_graph_memory_pool = graph_memory_pool
                else:
                    self.graph_memory_pool = graph_memory_pool

        def capture_graph_inner_tree(caches: List[torch.Tensor] = None) -> None:          
            # seq_len = self.speculate_length + 1  # TODO: change to tree.size
            is_multi_query_mode = False
            active_model = ActiveModel.DRAFT
            graph_runners = self.draft_graph_runners
            graph_memory_pool = self.draft_graph_memory_pool

            cu_layer_size = 0
            for step in self.speculate_tree.tree_layer:
                layer_size = self.speculate_tree.tree_layer[step]['layer_size']
                seq_len = layer_size
                cu_layer_size += layer_size

                # Prepare dummy inputs. These will be reused for all batch sizes.
                input_tokens = torch.zeros(max_batch_size,
                                        seq_len,
                                        dtype=torch.long,
                                        device="cuda")
                input_positions = torch.zeros(max_batch_size,
                                            seq_len,
                                            dtype=torch.long,
                                            device="cuda")
                slot_mapping = torch.empty(max_batch_size,
                                        seq_len,
                                        dtype=torch.long,
                                        device="cuda")
                slot_mapping.fill_(_PAD_SLOT_ID)
                context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
                block_tables = torch.from_numpy(self.graph_block_tables).cuda()
                tree_mask = torch.ones(seq_len, cu_layer_size, dtype=torch.bool).cuda()  

                with custom_all_reduce.capture():
                    for batch_size in reversed(batch_size_capture_list):
                        # Create dummy attn_metadata.
                        attn_metadata = self.attn_backend.make_metadata(
                            is_prompt=False,
                            slot_mapping=slot_mapping[:batch_size].view(-1),
                            prompt_lens=None,
                            prompt_lens_tensor=None,
                            num_prompt_tokens=0,
                            num_generation_tokens=0,
                            max_subquery_len=None,
                            max_context_len=self.max_context_len_to_capture,
                            max_prompt_len=None,
                            subquery_start_loc=None,
                            seq_start_loc=None,
                            context_lens=context_lens[:batch_size],
                            block_tables=block_tables[:batch_size],
                            use_cuda_graph=True,
                            kv_cache_dtype=self.kv_cache_dtype,
                            seq_len=seq_len,
                            is_multi_query_mode=is_multi_query_mode,
                            tree_mask = tree_mask
                        )

                        graph_runner = CUDAGraphRunner(self.draft_model)
                        with MarkActiveModel(active_model):
                            graph_runner.capture(
                                input_ids=input_tokens[:batch_size].view(-1),
                                positions=input_positions[:batch_size].view(-1),
                                kv_caches=caches,
                                attn_metadata=attn_metadata,
                                memory_pool=graph_memory_pool,
                            )

                        graph_memory_pool = graph_runner.graph.pool()
                        graph_runners[batch_size, step] = graph_runner

            self.draft_graph_memory_pool = graph_memory_pool

        def capture_graph_inner_tree_aug(caches: List[torch.Tensor] = None) -> None:     
            tree = self.speculate_tree     
            # seq_len = self.speculate_length + 1  # TODO: change to tree.size
            is_multi_query_mode = False
            active_model = ActiveModel.DRAFT
            graph_runners = self.draft_graph_runners
            graph_memory_pool = self.draft_graph_memory_pool

            draft_tokens = torch.zeros([max_batch_size, tree.size], dtype=torch.int64, device='cuda')
            slot_mapping = torch.empty(max_batch_size, tree.size, dtype=torch.long,device="cuda")
            slot_mapping.fill_(_PAD_SLOT_ID)
            
            context_lens = torch.ones(max_batch_size, dtype=torch.int64).cuda()
            positions = context_lens.view(-1, 1) + tree.positions.view(1, -1)

            block_tables = torch.from_numpy(self.graph_block_tables).cuda()

            cu_layer_size = 0
            for step in self.speculate_tree.tree_layer:
                layer_size = self.speculate_tree.tree_layer[step]['layer_size']
                seq_len = layer_size
                cu_layer_size += layer_size

                # Prepare dummy inputs. These will be reused for all batch sizes.
                # input_tokens = torch.zeros(max_batch_size,
                #                         seq_len,
                #                         dtype=torch.long,
                #                         device="cuda")
                # input_positions = torch.zeros(max_batch_size,
                #                             seq_len,
                #                             dtype=torch.long,
                #                             device="cuda")
                # slot_mapping = torch.empty(max_batch_size,
                #                         seq_len,
                #                         dtype=torch.long,
                #                         device="cuda")
                # slot_mapping.fill_(_PAD_SLOT_ID)

                # tree_mask = torch.ones(seq_len, cu_layer_size, dtype=torch.bool).cuda()  

                with custom_all_reduce.capture():
                    for batch_size in reversed(batch_size_capture_list):
                        # Create dummy attn_metadata.
                        attn_metadata = self.attn_backend.make_metadata(
                            is_prompt=False,
                            slot_mapping=slot_mapping[:batch_size],
                            prompt_lens=None,
                            prompt_lens_tensor=None,
                            num_prompt_tokens=0,
                            num_generation_tokens=0,
                            max_subquery_len=None,
                            max_context_len=self.max_context_len_to_capture,
                            max_prompt_len=None,
                            subquery_start_loc=None,
                            seq_start_loc=None,
                            context_lens=context_lens[:batch_size],
                            block_tables=block_tables[:batch_size],
                            use_cuda_graph=True,
                            kv_cache_dtype=self.kv_cache_dtype,
                            seq_len=seq_len,
                            is_multi_query_mode=is_multi_query_mode,
                            tree_mask = tree.attn_mask
                        )

                        graph_runner = DraftGraphRunner2D(self)
                        with MarkActiveModel(active_model):
                            graph_runner.capture(
                                draft_tokens=draft_tokens[:batch_size],
                                positions=positions[:batch_size],
                                kv_caches=caches,
                                attn_metadata=attn_metadata,
                                tree=self.speculate_tree,
                                batch_size=batch_size,
                                grow_step=step,
                                cu_idx=cu_layer_size,
                                layer_size=layer_size,
                                graph_memory_pool=graph_memory_pool,
                            )

                        graph_memory_pool = graph_runner.graph.pool()
                        graph_runners[batch_size, step] = graph_runner

            self.draft_graph_memory_pool = graph_memory_pool

        def capture_graph_tree_verify(
                draft_kv_caches,
                target_kv_caches,
                graph_memory_pool=None):
            tree = self.speculate_tree
            if tree is None:    
                return
                
            vocab_size = self.model.model.vocab_size
            first_successor_idx = tree.successor_tail_idx[0].item()
            block_size = draft_kv_caches[0][0].size(1)
            block_tables = torch.from_numpy(self.graph_block_tables).to(torch.int64).cuda()
            for batch_size in reversed(batch_size_capture_list):
                probs = torch.ones(batch_size, tree.input_size, vocab_size, dtype=torch.float32, device="cuda")
                probs = probs / probs.sum(dim=-1, keepdim=True)
                draft_tokens = torch.zeros(batch_size, tree.size, dtype=torch.int64, device='cuda')
                prefix_length = torch.zeros(batch_size, dtype=torch.int64, device='cuda')

                graph_runner = GraphRunner(self.verification)
                graph_memory_pool = graph_runner.capture(
                    probs, 
                    draft_tokens, 
                    tree, 
                    first_successor_idx, 
                    prefix_length,
                    block_tables[:batch_size],
                    block_size,
                    draft_kv_caches,
                    target_kv_caches,
                    graph_memory_pool)
                self.tree_verification_graph_runners[batch_size] = graph_runner
                
            self.graph_memory_pool = graph_memory_pool


        # 1. Capture draft model
        if not self.speculate_tree:
            capture_graph_inner(is_draft_model=True, caches=draft_kv_caches)
        else:
            capture_graph_inner_tree_aug(caches=draft_kv_caches)
            # pass
        # 2. Capture target model
        capture_graph_inner(is_draft_model=False, caches=kv_caches)

        # 3. Capture tree verify
        capture_graph_tree_verify(draft_kv_caches, kv_caches)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")


    def _prepare_speculate_decode_2d(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        is_multi_query_mode: bool = False, tree = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, 'SpeculateAttnMetadata', List[int],
               List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        seq_data_dict: Dict[int, SequenceData] = {}

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1
            seq_data_dict.update(seq_group_metadata.seq_data)

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_group_metadata.seq_data:
                seq_data = seq_group_metadata.seq_data[seq_id]

                generation_tokens = [seq_data.get_last_token_id()
                                     ] + [0] * (tree.size - 1)  # last_token is the tree root

                input_tokens.append(generation_tokens)
                #: prompt_len: {11, 9} + generated_tkn_lens + spec_len or 1
                # including the input token
                prefix_len = seq_data.get_len()
                seq_len = prefix_len + (tree.size - 1)  # last_token is the tree root

                # TODO: context_len usage check
                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                # Tree position
                # positions = prefix_len - 1 + tree_depth  # last_token is the tree root
                positions = prefix_len - 1  # last_token is the tree root excluded
                input_positions.append(positions)

                block_table = seq_group_metadata.block_tables[seq_id]

                # TODO: check slots usage
                slots = []  # slots to store the kv_cache for the input
                for idx_list in tree.grow_map['roots']:
                    for idx in idx_list:
                        pos = prefix_len - 1 + idx
                        block_number = block_table[pos // self.block_size]
                        block_offset = pos % self.block_size
                        slot = block_number * self.block_size + block_offset
                        slots.append(slot)
                slot_mapping.append(slots)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        # vLLM uses cuda graph only for decoding requests.
        # See `capture_model` API for more details.
        batch_size = len(input_tokens)
        # record the real batch size for cudagraph in case of padding
        max_context_len = max(context_lens)
        use_captured_graph = (
            not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_context_len <= self.max_context_len_to_capture)
        # use_captured_graph = False

        graph_batch_size = batch_size
        if use_captured_graph:
            # Pad the input tokens, positions, and slot mapping to match the
            # batch size of the captured graph.
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append([])
                input_positions.append([])
                slot_mapping.append([])
                context_lens.append(0)
                block_tables.append([])
            batch_size = graph_batch_size

        # Convert list to padded tensor
        max_len = max([len(t) for t in input_tokens])
        input_tokens = make_tensor_with_pad(input_tokens,
                                            max_len=max_len,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)
        slot_mapping = make_tensor_with_pad(slot_mapping,
                                            max_len=max_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.long,
                                            device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.long,
                                    device=self.device)

        tree_mask = tree.attn_mask

        if use_captured_graph:
            # When using cuda-graph all these tensors should be
            # padded.
            assert context_lens.shape[0] == input_tokens.shape[0]
            # assert context_lens.shape[0] == input_positions.shape[0]
            assert context_lens.shape[0] == slot_mapping.shape[0]

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, dtype=torch.long, device=self.device)

        else:
            max_block_table_len = max(
                len(block_table) for block_table in block_tables)
            block_tables = make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.long,
                device=self.device,
            )


        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            prompt_lens=None,
            prompt_lens_tensor=None,
            num_prompt_tokens=0,
            num_generation_tokens=len(input_tokens),
            max_subquery_len=None,
            max_context_len=max_context_len,
            max_prompt_len=None,
            subquery_start_loc=None,
            seq_start_loc=None,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            kv_cache_dtype=self.kv_cache_dtype,
            seq_len=input_tokens.shape[1],
            is_multi_query_mode=is_multi_query_mode,
            tree_mask = tree_mask,
        )
        sampling_metadata = SpeculativeSamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data_dict,
            prompt_lens=None,
            selected_token_indices=None,
            categorized_sample_indices=None,
            generators=None,
            is_multi_query_mode=is_multi_query_mode,
            speculate_tree=tree,
        )
        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata)


    def _prepare_speculate_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        is_multi_query_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, 'SpeculateAttnMetadata',
               'SpeculativeSamplingMetadata', int]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        seq_data_dict: Dict[int, SequenceData] = {}

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1
            seq_data_dict.update(seq_group_metadata.seq_data)

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_group_metadata.seq_data:
                seq_data = seq_group_metadata.seq_data[seq_id]

                num_generation_tokens = self.speculate_length + 1
                generation_tokens = [seq_data.get_last_token_id()
                                     ] + [0] * self.speculate_length

                input_tokens.append(generation_tokens)
                # prompt_len: {11, 9} + generated_tkn_lens + spec_len or 1
                seq_len = seq_data.get_len() + self.speculate_length

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                first_position = seq_len - num_generation_tokens
                positions = [
                    first_position + offset
                    for offset in range(num_generation_tokens)
                ]
                input_positions.append(positions)

                block_table = seq_group_metadata.block_tables[seq_id]

                slots = []
                for position in positions:
                    block_number = block_table[position // self.block_size]
                    block_offset = position % self.block_size
                    slot = block_number * self.block_size + block_offset
                    slots.append(slot)
                slot_mapping.append(slots)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        # vLLM uses cuda graph only for decoding requests.
        # See `capture_model` API for more details.
        batch_size = len(input_tokens)
        # record the real batch size for cudagraph in case of padding
        real_batch_size = batch_size
        max_context_len = max(context_lens)
        use_captured_graph = (
            not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_context_len <= self.max_context_len_to_capture)
        # use_captured_graph = False
        if use_captured_graph:
            # Pad the input tokens, positions, and slot mapping to match the
            # batch size of the captured graph.
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append([])
                input_positions.append([])
                slot_mapping.append([])
                context_lens.append(0)
                block_tables.append([])
            batch_size = graph_batch_size

        max_len = max([len(t) for t in input_tokens])
        input_tokens = make_tensor_with_pad(input_tokens,
                                            max_len=max_len,
                                            pad=0,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               max_len=max_len,
                                               pad=0,
                                               dtype=torch.long,
                                               device=self.device)
        slot_mapping = make_tensor_with_pad(slot_mapping,
                                            max_len=max_len,
                                            pad=_PAD_SLOT_ID,
                                            dtype=torch.long,
                                            device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)

        if use_captured_graph:
            # When using cuda-graph all these tensors should be
            # padded.
            assert context_lens.shape[0] == input_tokens.shape[0]
            assert context_lens.shape[0] == input_positions.shape[0]
            assert context_lens.shape[0] == slot_mapping.shape[0]

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device=self.device)
        else:
            max_block_table_len = max(
                len(block_table) for block_table in block_tables)
            block_tables = make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            prompt_lens=None,
            prompt_lens_tensor=None,
            num_prompt_tokens=0,
            num_generation_tokens=len(input_tokens),
            max_subquery_len=None,
            max_context_len=max_context_len,
            max_prompt_len=None,
            subquery_start_loc=None,
            seq_start_loc=None,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            kv_cache_dtype=self.kv_cache_dtype,
            seq_len=input_tokens.shape[1],
            is_multi_query_mode=is_multi_query_mode,
        )
        sampling_metadata = SpeculativeSamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data_dict,
            prompt_lens=None,
            selected_token_indices=None,
            categorized_sample_indices=None,
            generators=None,
            is_multi_query_mode=is_multi_query_mode,
            speculate_length=self.speculate_length,
        )
        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, real_batch_size)


    def prepare_speculate_decode_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ):
        input_tokens, input_positions, attn_metadata, sampling_metadata = self._prepare_speculate_decode(
            seq_group_metadata_list)
        # # Broadcast the metadata.
        # metadata_dict = {
        #     "input_tokens": input_tokens,
        #     "input_positions": input_positions,
        # }
        # metadata_dict.update(attn_metadata.asdict_zerocopy())
        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata)

    def fast_greedy_sample(self, model: torch.nn.Module,
                           hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: We always use greedy sampling for the draft model during
        # speculative decoding. In this way, we can greatly reduce the
        # sampling overhead and avoid storing draft token's logits.
        embedding_bias = None
        if hasattr(model, "lm_head"):
            lm_head_weight = model.lm_head.weight
            embedding_bias = model.lm_head.bias
        elif hasattr(model, "lm_head_weight"):
            lm_head_weight = model.lm_head_weight
        elif hasattr(model, "embed_out"):
            lm_head_weight = model.embed_out.weight
        elif hasattr(model, "embed_tokens"):
            lm_head_weight = model.embed_tokens.weight
        else:
            raise RuntimeError("Unsupported draft model")
        assert len(hidden_states.shape
                   ) == 2, "hidden_states must have shape [bs*seq_len, dim]"
        logits = torch.matmul(hidden_states, lm_head_weight.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_all_gather(logits)
        # TODO: This will not be right with lora.
        logits = logits[..., :model.config.vocab_size]
        next_tokens = torch.argmax(logits, dim=-1)
        return next_tokens

    def apply_model_head(self, model: torch.nn.Module,
                           hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: We always use greedy sampling for the draft model during
        # speculative decoding. In this way, we can greatly reduce the
        # sampling overhead and avoid storing draft token's logits.
        embedding_bias = None
        if hasattr(model, "lm_head"):
            lm_head_weight = model.lm_head.weight
            embedding_bias = model.lm_head.bias
        elif hasattr(model, "lm_head_weight"):
            lm_head_weight = model.lm_head_weight
        elif hasattr(model, "embed_out"):
            lm_head_weight = model.embed_out.weight
        elif hasattr(model, "embed_tokens"):
            lm_head_weight = model.embed_tokens.weight
        else:
            raise RuntimeError("Unsupported draft model")
        assert len(hidden_states.shape
                   ) == 2, "hidden_states must have shape [bs*seq_len, dim]"
        logits = torch.matmul(hidden_states, lm_head_weight.t())
        if embedding_bias is not None:
            logits += embedding_bias
        logits = tensor_model_parallel_all_gather(logits)
        logits = logits[..., :model.config.vocab_size]
        return logits

    def speculate_prefill_step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        draft_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> SpeculativeSpeculateOutput:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)
        attn_metadata.slot_mapping = attn_metadata.slot_mapping.view(-1)
        input_tokens_1d = input_tokens.view(-1)
        input_positions_1d = input_positions.view(-1)
        # 1. Draft model prefill.
        # The purpose of this step is to fill draft model's kv cache
        with MarkActiveModel(ActiveModel.DRAFT):
            hidden_states = self.draft_model(
                input_ids=input_tokens_1d,
                positions=input_positions_1d,
                kv_caches=draft_kv_caches,
                attn_metadata=attn_metadata,
            )
        # 2. Target model prefill
        with MarkActiveModel(ActiveModel.TARGET):
            # Execute the model.
            hidden_states = self.model(
                input_ids=input_tokens_1d,
                positions=input_positions_1d,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )
            # Compute the logits.
            logits = self.model.compute_logits(hidden_states,
                                               sampling_metadata)
            # Only perform sampling in the driver worker.
            if not sampling_metadata.perform_sampling:
                return None
            # Sample the next token.
            output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        speculate_outputs: SpeculativeSpeculateOutput = []
        for seq_group_output in output:
            samples = seq_group_output.samples
            assert len(
                samples
            ) == 1, "Speculative decoding only allows one seq per seq group."
            sample = samples[0]
            seq_id = sample.parent_seq_id
            token_id = sample.output_token
            logprobs_dict = sample.logprobs
            speculate_outputs.append(
                SpeculativeSpeculateSequenceGroupOutput(
                    seq_id, 
                    [token_id], 
                    [logprobs_dict],
                    0,
                    seq_group_output.prompt_logprobs))
        return speculate_outputs


    def drafting(self, 
                 draft_tokens, 
                 slot_mapping,
                 tree_positions,
                 tree_mask,
                 draft_kv_caches, 
                 attn_metadata, 
                 tree, batch_size, grow_step, cu_idx, layer_size):
        # Warning: without the .contiguous() the decoding will hang.
        # Somehow .contiguous() and .view(-1) is suddenly supported by torch graph capture
        input_tokens_1d = draft_tokens[:, cu_idx - layer_size: cu_idx].contiguous().view(-1)
        attn_metadata.slot_mapping = slot_mapping[:, cu_idx - layer_size : cu_idx].contiguous().view(-1)

        query_start = cu_idx - layer_size
        query_end = cu_idx
        input_positions_1d = tree_positions[:, query_start: query_end].contiguous().view(-1)
        attn_metadata.tree_mask = tree_mask[query_start: query_end, :cu_idx].contiguous()

        # [bs, layer_size, hid_size]
        draft_hidden_states = self.draft_model(
            input_ids=input_tokens_1d,
            positions=input_positions_1d,
            kv_caches=draft_kv_caches,
            attn_metadata=attn_metadata
        )

        layer_size = len(tree.grow_map['roots'][grow_step])
        draft_output = self.apply_model_head(self.draft_model, draft_hidden_states)
        draft_output = draft_output.view(batch_size, layer_size, -1)

        # Outside With Mark
        # Generate children tokens
        next_layer_size = sum(tree.grow_map['branches'][grow_step])
        # [bs, layer_size * max_sampling]
        beam_tokens = tree.sampling_callables[grow_step](draft_output).view(batch_size, -1)
        # [bs, next_layer_size]
        next_layer_tokens = beam_tokens[:, tree.sample_gather_indices[grow_step]]
        # assert next_layer_size == next_layer_tokens.shape[-1]
        draft_tokens[:, cu_idx: cu_idx + next_layer_size] = next_layer_tokens
        return draft_tokens


    @torch.inference_mode()
    def speculate_decode_step_2d(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        target_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        draft_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        step=0
    ) -> SpeculativeSpeculateOutput:
        real_batch_size = len(seq_group_metadata_list)
        tree: Tree = self.speculate_tree
        if not tree:
            raise Exception(f"Need speculate tree input")

        # Input
        input_tokens, input_positions, attn_metadata, sampling_metadata = \
            self._prepare_speculate_decode_2d(seq_group_metadata_list, tree=tree)

        graph_batch_size = input_tokens.shape[0]


        # [idx_list[i] * ith_max_branch(beam_size)] which is indexed by gather_list[i]
        # new_tokens_set = self.sampling_callables[grow_step](self.draft_logits[idx_list])

        # --- Preallocation ---
        batch_size = graph_batch_size
        dtype = torch.float16
        device = f'cuda:{dist.get_rank()}' if dist.is_initialized() else 'cpu'
        vocab_size = 32000

        tree_size = tree.size
        tree_depth = tree.depth
        #: [0, L-1] are target model inputs, [1, L] are tokens to verify
        #: [0, tree_size]
        draft_tokens = torch.zeros([graph_batch_size, tree_size],
                                dtype=torch.int64, device=device)
        draft_tokens[:, 0] = input_tokens[:, 0]  # root

        # exclude the current input token
        prefix_length = attn_metadata.context_lens - tree_size
        # max_context_len = attn_metadata.max_context_len - tree_size  # exclude the tree root
        # [bs, tree_size]
        tree_positions = prefix_length.view(-1, 1) + tree.positions.view(1, -1)
        slot_mapping = attn_metadata.slot_mapping
        # [q_seqlen, kv_seqlen]
        tree_mask = attn_metadata.tree_mask

        # 1. ======== Drafting ========
        cu_idx = 0  # the already processed input_length
        # cu_context_lens = torch.cumsum(prefix_length, dim=0)  # -1 is to exclude the tree root
        # cu_context_lens = torch.cat((torch.tensor([0], device=cu_context_lens.device, 
        # dtype=cu_context_lens.dtype), cu_context_lens))
        context_lens = prefix_length
        for grow_step in range(tree_depth - 1):
            # grow_step = 0, cu_index = 0, layer_size = 1
            # grow_step = 1, cu_index = 1, layer_size = 16
            layer_size = len(tree.grow_map['roots'][grow_step])
            cu_idx += layer_size
            # cu_context_lens = cu_add(cu_context_lens, layer_size)
            context_lens = context_lens + layer_size
            with MarkActiveModel(ActiveModel.DRAFT):
                model_executable = self.draft_model
                # Multi query drafting. input_ids: [bs, layer_size=16]
                attn_metadata.seq_len = layer_size
                attn_metadata.context_lens = context_lens

                if attn_metadata.use_cuda_graph:
                    draft_tokens = self.draft_graph_runners[
                        graph_batch_size, grow_step](
                            draft_tokens, 
                            slot_mapping,
                            tree_positions,
                            tree_mask,
                            attn_metadata)
                else:
                    draft_tokens = self.drafting(
                        draft_tokens, 
                        slot_mapping,
                        tree_positions,
                        tree_mask,
                        draft_kv_caches, 
                        attn_metadata, 
                        tree, batch_size, grow_step, cu_idx, layer_size)

        dtype = self.model_config.dtype

        # 2. ======== Run target model ========
        last_layer_size = len(tree.grow_map['roots'][-1])
        input_size = tree_size - last_layer_size

        sampling_metadata.input_token_ids = draft_tokens[:real_batch_size, :input_size]
        sampling_metadata.is_multi_query_mode = True

        logits = torch.full((graph_batch_size, tree_size, vocab_size), torch.finfo(dtype).min, dtype=dtype, device=device)

        with MarkActiveModel(ActiveModel.TARGET):
            model_executable = self.model
            attn_metadata.is_multi_query_mode = True
            attn_metadata.seq_len = input_size  # including the root
            # [bs, seq_len]
            input_tokens_1d = draft_tokens[:, :input_size].reshape(-1)
            input_positions_1d = tree_positions[:, :input_size].reshape(-1)
            attn_metadata.tree_mask = tree_mask[:input_size, :input_size]
            attn_metadata.slot_mapping = slot_mapping[:, :input_size].reshape(-1)
            attn_metadata.context_lens = prefix_length + input_size

            if attn_metadata.use_cuda_graph:
                graph_runners = self.graph_runners
                # For speculative decoding, cudagraph is only used in the evaluation stage
                # for target model.
                model_executable = graph_runners[graph_batch_size]
            # Execute the model.        
            # [batch_size * input_size, dim]
            hidden_states = model_executable(
                input_ids=input_tokens_1d.to(device),
                positions=input_positions_1d.to(device),
                kv_caches=target_kv_caches,
                attn_metadata=attn_metadata,
            )
            hidden_states_cudagraph = hidden_states.view(batch_size, input_size, -1)

            hidden_states = hidden_states_cudagraph

            # Compute the logits. [batch_size, input_size, dim]
            logits[:, :input_size, :] = self.model.compute_logits(hidden_states, sampling_metadata)
        
        # torch.allclose(logits, logits, rtol=1e-3, atol=1e-3)     
        # Inside `ModelRunner.execute_model()`
        sampler = self.model.sampler
        # only perform sampling in the driver worker
        if not sampling_metadata.perform_sampling:
            return None
        # Inside `VllmSample.sample()`
        # Prepare sampling tensors with pinned memory to avoid blocking.
        sampling_tensors, sampling_flags = sampler.get_sampling_tensors(sampling_metadata, 
                                                                        logits.shape[-1], 
                                                                        logits.device,
                                                                        logits.dtype)
        # torch.allclose(logits, logits, rtol=1e-3, atol=1e-3)     
        logits[:real_batch_size, :input_size] = sampler.process_logits(
            logits[:real_batch_size, :input_size, :], 
            sampling_metadata, 
            sampling_tensors, 
            sampling_flags)
        # [bs, input_size, vocab_size]
        logits = logits[:, :input_size]

        # logits_ben = logits.clone()
        ## Debug
        if _debug:
            print0(f"cnt={step}, tree_size = {tree.size}")
            print0(f"draft_token\n{draft_tokens}, {draft_tokens.numel()}")

            print0(f"target\n{logits.argmax(dim=-1)}")


        # # Regulation check
        # logits[real_batch_size:] = 0.1
        # probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # assert not torch.isnan(probs).any()
        # assert torch.allclose(probs.sum(dim=-1), torch.ones(*probs.shape[:-1], device=probs.device), rtol=1e-3, atol=1e-3)       
        # torch.allclose(logits, logits, rtol=1e-3, atol=1e-3)     

        # ======== Utilities ========
        def scan_batch(output_acc_idx, acc_lengths, prefix_lengths, block_table, block_size):
            """
            Args:
                output_acc_idx:
                    [bs, tree_depth] excluding the root. But the value starts with the root, i.e. root_idx = 0
                acc_lengths:
                    the accepted length excluding the root and the extra token
                prefix_lengths:
                    prefix lengths exclude the root, thus prefix_length is the root index
                block_table: [bs, length]
            Return:
                slot_mapping: the destination slots in the consolidation
                slot_source: the source slots in the consolidation
            """
            assert block_size == 16
            size = acc_lengths.sum().item()  # the size to consolidate
            slot_source = torch.zeros(size, dtype=torch.int64, device=acc_lengths.device)
            slot_mapping = torch.zeros(size, dtype=torch.int64, device=acc_lengths.device)

            def get_slots(token_indices, batch_idx):
                indices_blk_tbl = token_indices // block_size
                indices_offset = token_indices % block_size

                block_numbers = block_table[batch_idx, indices_blk_tbl]
                slots = block_numbers * block_size + indices_offset
                return slots

            idx = 0
            for b in range(acc_lengths.size(0)):
                # Get the consolidate indices
                acc_len = acc_lengths[b].item()
                prefix_len = prefix_lengths[b].item()
                # if acc_len == 0: continue

                token_indices = prefix_len + output_acc_idx[b, :acc_len]  # values of output_acc_idx starts from 1 since the root is excluded
                slot_source[idx: idx + acc_len] = get_slots(token_indices, b)

                token_indices = torch.arange(prefix_len + 1, prefix_len + 1 + acc_len, device=acc_lengths.device, dtype=torch.int64)  # root occupies a position
                slot_mapping[idx: idx + acc_len] = get_slots(token_indices, b)

                idx += acc_len

            return slot_mapping, slot_source

        def scan_batch2(output_tokens, output_acc_idx, target_probs_eval, acc_lengths, seq_id_bonus_bool):
            """
            Args:
                output_tokens: [bs, tree_depth] excluding the root.
            """
            speculate_outputs: SpeculativeSpeculateOutput = []
            # seq_id_bonus_set = set(seq_id_bonus.tolist())
            for b, (seq_ids, sampling_params) in enumerate(sampling_metadata.seq_groups):
                acc_len = acc_lengths[b].item()
                 # SpeculativeSpeculateOutput
                assert len(seq_ids) == 1
                num_generated_tokens = acc_len + 1
                token_list = output_tokens[b, :num_generated_tokens].tolist()
                # print0(f'token_list {token_list}')
                acc_idx = output_acc_idx[b, :num_generated_tokens]
                output_token_logprobs = [{token_id: Logprob(logprob=logprob)} for token_id, logprob in zip(token_list, target_probs_eval[b, acc_idx].tolist())]
                speculate_outputs.append(
                    SpeculativeSpeculateSequenceGroupOutput(seq_ids[0],
                                                            token_list,
                                                            output_token_logprobs,
                                                            acc_len + (1 if seq_id_bonus_bool[b].item() else 0),
                                                            prompt_logprobs=None))

            return speculate_outputs           

        def consolidate_cache(key_cache, value_cache, output_token_idx, acc_lengths, prefix_length, block_tables):
            slots_source = []
            slots_mapping = []
            block_size = key_cache.size(1)
            num_heads = value_cache.size(2)
            head_size = value_cache.size(3)
            num_token = output_token_idx.numel()
            gpu_block_stride = num_heads * head_size
            key = torch.empty(num_token, num_heads, head_size, dtype=key_cache.dtype, device=key_cache.device)
            value = torch.empty(num_token, num_heads, head_size, dtype=key_cache.dtype, device=key_cache.device)
            
            # Fetch
            for block_idx in range(num_token):
                entry_idx = block_idx
                batch_idx = entry_idx // output_token_idx.stride(0)
                seq_idx = entry_idx % output_token_idx.stride(0)
                if (acc_lengths[batch_idx] <= seq_idx):
                    continue
                token_idx = (prefix_length[batch_idx] + output_token_idx.view(-1)[entry_idx]).item()

                block_table_idx = token_idx // block_size
                offset = token_idx % block_size
                block_number = block_tables.view(-1)[batch_idx * block_tables.stride(0) + block_table_idx].item()
                slot_idx = block_number * block_size + offset

                slots_source.append(slot_idx)
                for i in range(gpu_block_stride):
                    #// Fetch the cache according to slot_idx
                    key_elem = key_cache.view(-1)[slot_idx * key_cache.stride(1) + i]
                    value_elem = value_cache.view(-1)[slot_idx * value_cache.stride(1) + i]

                    #// Put the cache in the storage according to entry_idx
                    key.view(-1)[entry_idx * key.stride(0) + i] = key_elem
                    value.view(-1)[entry_idx * value.stride(0) + i] = value_elem

            torch.cuda.synchronize()

            # Put
            for block_idx in range(num_token):
                entry_idx = block_idx
                batch_idx = entry_idx // output_token_idx.stride(0)
                seq_idx = entry_idx % output_token_idx.stride(0)
                if (acc_lengths[batch_idx] <= seq_idx):
                    continue
                token_idx = prefix_length[batch_idx].item() + 1 + seq_idx

                # // Get the slot_idx of token_idx
                block_table_idx = token_idx // block_size
                offset = token_idx % block_size
                block_number = block_tables.view(-1)[batch_idx * block_tables.stride(0)+ block_table_idx].item()
                slot_idx = block_number * block_size + offset
                
                slots_mapping.append(slot_idx)

                for i in range(gpu_block_stride):
                    # // Fetch the entry from key or value according to slot_idx
                    key_elem = key.view(-1)[entry_idx * key.stride(0) + i]
                    value_elem = value.view(-1)[entry_idx * value.stride(0) + i]

                    # // Put the cache in the storage according to entry_idx
                    key_cache.view(-1)[slot_idx * key_cache.stride(1) + i] = key_elem
                    value_cache.view(-1)[slot_idx * value_cache.stride(1) + i] = value_elem
            
            return slots_source, slots_mapping

        def consolidate_kv_cache(kv_cache_list, slot_mapping, slot_source):
            if slot_mapping.numel() == 0:
                assert slot_source.numel() == 0, f"slot_source and slot_mapping don't match"
                return
            for kv_cache in kv_cache_list:
                for key_cache, val_cache in kv_cache:
                    rubikon_cache_ops.consolidate_cache_flash_layout(
                        key_cache,
                        val_cache,
                        slot_source,
                        slot_mapping,
                        'auto',
                    )
       
        # 3. ======== Verification ========
        torch.manual_seed(2024)
        first_successor_idx = tree.successor_tail_idx[0].item()
        verify_executor = lambda x1, x2, x3, x4, x5, x6: self.verification(
            x1, # probs
            x2, # draft_tokens
            tree.parent_group_idx, 
            tree.bin_filter, 
            tree.parent_idx, 
            tree_size, tree_depth, 
            tree.successor_tail_idx, 
            tree.successor_tail_idx_stablized, first_successor_idx, batch_size,
            x3, # prefix_length
            x4, # attn_metadata.block_tables,
            draft_kv_caches[0].size(2), 
            x5, # draft_kv_caches, 
            x6) # target_kv_caches)

        if attn_metadata.use_cuda_graph:
            verify_executor = self.tree_verification_graph_runners[batch_size]

        output_tokens, output_token_idx, acc_lengths, probs_eval, seq_id_bonus_bool = verify_executor(
            logits, draft_tokens, 
            prefix_length, attn_metadata.block_tables, draft_kv_caches, target_kv_caches)

        # for kv_cache in [draft_kv_caches, target_kv_caches]:
        #     for key_cache, val_cache in kv_cache:
        #         rubikon_cache_ops.consolidate_cache_flash_layout(
        #             key_cache,
        #             val_cache,
        #             slot_source_cu,
        #             slot_mapping_cu,
        #             'auto',
        #         )

        # # torch.allclose(logits, logits, rtol=1e-3, atol=1e-3)  
        
        # seq_id_bonus = torch.nonzero(seq_id_bonus_bool).view(-1)
        
        speculate_outputs = scan_batch2(output_tokens, output_token_idx, probs_eval, acc_lengths, seq_id_bonus_bool)
        # torch.allclose(logits, logits, rtol=1e-3, atol=1e-3)  
         
        # def filter_slot_idx(idx_cu):
        #     out_list = []
        #     for i, idx in enumerate(idx_cu.view(-1).tolist()):
        #         if idx < 0: continue
        #         out_list.append(idx)
        #     return torch.tensor(out_list, dtype=torch.int64, device=idx_cu.device)
        # slot_mapping_idx = filter_slot_idx(slot_mapping_cu)
        # slot_source_idx = filter_slot_idx(slot_source_cu)   
        
        # ======= Delimiter line. Above: opt. Below: primitive ========

        # # 4.5 ----- Verify_bench -----
        # output_tokens_ben, output_token_idx_ben, acc_lengths_ben, probs_eval_ben, seq_id_bonus_ben = self.verification_ben(logits_ben, draft_tokens, tree)

        # # 5.5 ======== Consolidate the kv_cache according to acc_path ========      
        # slot_mapping, slot_source = scan_batch(output_token_idx_ben, acc_lengths_ben, prefix_length, attn_metadata.block_tables, target_kv_caches[0][0].shape[1])

        # consolidate_kv_cache([draft_kv_caches, target_kv_caches], slot_mapping, slot_source)
        # speculate_outputs_ben = scan_batch2(output_tokens_ben, output_token_idx_ben, probs_eval_ben, acc_lengths_ben, seq_id_bonus_ben)
        
        # if _debug:
        #     print0(f"acc_len {acc_lengths_ben[0].item() + (1 if 0 in set(seq_id_bonus_ben.tolist()) else 0)} acc_list = {output_token_idx_ben[0, :acc_lengths_ben[0].item()]}")
        #     print0(f"valid tokens\n{output_tokens_ben[0, :acc_lengths_ben[0].item()+1]}")
        #     print0("-----------------------------")

        # torch.testing.assert_close(slot_mapping_idx, slot_mapping)
        # torch.testing.assert_close(slot_source_idx, slot_source)
        # torch.testing.assert_close(acc_lengths, acc_lengths_ben)
        # torch.testing.assert_close(seq_id_bonus, seq_id_bonus_ben)
        # torch.testing.assert_close(output_tokens[0, :acc_lengths[0].item() + 1], output_tokens_ben[0, :acc_lengths_ben[0].item() + 1])

        return speculate_outputs


    @staticmethod
    def verification_ben(logits, draft_tokens, tree):
        """
        Input:
            logits: [bs, input_size, vocab_size]
            draft_tokens: [bs, tree_size]
        Return:
            output_tokens: [bs, tree_depth] excluding the root.
            output_acc_idx:
                [bs, tree_depth] excluding the root. But the value starts with the root, i.e. root_idx = 0
            acc_lengths:
                the accepted length excluding the root and the extra token. acc_len = len(acc_list)
            seq_id_bonus: 
                the seq_id whose last layer is accepted. Thus acc_len - 1.
            output_tokens, output_token_idx, acc_lengths, probs_eval, seq_id_bonus
        """
        bs, input_size, _ = logits.shape
        _, tree_size = draft_tokens.shape
        tree_depth = tree.depth

        device = logits.device
        output_tokens = torch.zeros(bs, tree_depth, dtype=torch.int64, device=device)
        output_acc_idx = torch.zeros(bs, tree_depth, dtype=torch.int64, device=device)
        acc_lengths = torch.full((bs,), -1, dtype=torch.int64, device=device)
        seq_id_bonus = []

        # [bs, input_size]
        target_token = logits.argmax(dim=-1)
        probs_eval = torch.zeros_like(target_token, dtype=torch.float16, device=device)

        def recur(node, acc_list, b):
            # Find the acc_idx in the children of node
            # acc_list is the accepted draft_token indices (excluding root). len(acc_list) is the acc_len. target_token is kept up to acc_len + 1. The bonus layer is automatically treated as not accepted, which is implemented by acc_len - 1 on seq_id_bonus. 
            if tree_depth - 1 == len(acc_list):
                # node will be beyond input_size
                return

            next_token = target_token[b, node].item()
            for child in tree.grow_map['Successors'][node]:
                if draft_tokens[b, child].item() == next_token:
                    acc_list.append(child)
                    recur(child, acc_list, b)
                    break
            else:
                # The acc_len can be inferred from len(acc_list)
                pass
        
        for b in range(bs):
            acc_list = []
            recur(0, acc_list, b)
            # post process
            acc_len = len(acc_list)
            if acc_len == tree_depth - 1:
                seq_id_bonus.append(b)
                acc_len -= 1
            # output_tokens[b, :acc_len + 1] = target_token[b, :acc_len + 1]
            output_acc_idx[b, :acc_len] = torch.tensor(acc_list[:acc_len], device=device)  # The bonus accepted token is discarded due to -1 above
            output_tokens[b, 0] = target_token[b, 0] # the next token of root is always output
            output_tokens[b, 1 : acc_len + 1] = target_token[b, output_acc_idx[b, :acc_len]]
            acc_lengths[b] = acc_len
        
        seq_id_bonus = torch.tensor(seq_id_bonus, device=device, dtype=torch.int64)
        return output_tokens, output_acc_idx, acc_lengths, probs_eval, seq_id_bonus
    

    @staticmethod
    def verification(logits, draft_tokens,
                     parent_group_idx, bin_filter, parent_idx, tree_size, tree_depth, 
                     successor_tail_idx, successor_tail_idx_stablized, first_successor_idx, batch_size,
                     prefix_length, 
                     block_tables,
                     block_size, 
                     draft_kv_caches: List[torch.Tensor], 
                     target_kv_caches: List[torch.Tensor]):
        """
        probs: [bs, input_size, vocab_size]
        draft_tokens: [bs, tree_size]
        """
        # 3. ======== Verification ========
        # ======================
        # Definitions
        # SIBLING GROUP: siblings of the same parent. 
        # E.g.  [(0), (1,..., 16), (17, ..., 26), (27,..., 30), (31, 32) ...]
        # Sizes: 1    16           10             4             2
        # SIBLING GROUP INDEX: represented by its last member's index
        # E.g. [0, 16, 26, 30, 32, ...]
        # ======================
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)

        # Get probs_residual, probs_residual_eval, and probs
        row_idx = torch.arange(batch_size, device=probs.device, dtype=torch.int64)
        # batch_size = real_batch_size
        # draft_tokens: [bs, tree_size] 
        # probs: [bs, tree_size, vocab]
        # x0 | x1, ..., x16 | x17 ... x26 | ... | last_layer
        vocab_size = probs.shape[-1]
        # parent_idx: [0x16, 1x10, 2x4, ...]
        assert parent_idx.shape[0] == draft_tokens.shape[1] - 1 == tree_size - 1
        #: [bs, tree_size - 1]: [(x1, ..., x16), (x17,..., x26), ...]; to index a tree_size logits/probs
        token_flatten_idx = parent_idx * vocab_size + draft_tokens[:, 1:]
        #: [bs, tree_size - 1]: [(x1, ..., x16), (x17,..., x26), ...]
        probs_eval = torch.gather(probs.view(batch_size, -1), 1, token_flatten_idx)
        #: [bs, tree_size]: [(x0), (x1, ..., x16), (x17,..., x26), ...]
        probs_eval = torch.nn.functional.pad(probs_eval, (1, 0), mode='constant', value=0)

        #: [bs, tree_size]: [0, p(x1, ..., x16), p(x17,..., x26), p(x27,..., x30), ...]
        prefix_sum_probs = torch.cumsum(probs_eval, dim=-1)
        assert prefix_sum_probs.shape[1] == parent_group_idx.shape[0] == tree_size
        # parent_group_idx: [0, 0x16, 16x10, 26x4, ...].
        #: [bs, tree_size]
        prefix_sum_probs_per_group = prefix_sum_probs - prefix_sum_probs[:, parent_group_idx]

        # successor_tail_idx: [(16), (26, 30, 32, ..., )]
        input_size = probs.shape[1]
        #: [bs, tree_size, vocab_size]
        probs_residual = probs / (1 - prefix_sum_probs_per_group[:, successor_tail_idx[:input_size]]).unsqueeze_(2)
        rubikon_cache_ops.set_entry_selective(probs_residual, token_flatten_idx)
        # expanded_row_idx = row_idx.unsqueeze(1).expand(-1, token_flatten_idx.size(1))
        # probs_residual.view(batch_size, -1)[expanded_row_idx, token_flatten_idx] = 0

        # [bs, tree_size]
        # assert probs_eval.shape[1] == tree_size == prefix_sum_probs_per_group.shape[1]
        probs_residual_eval = probs_eval / (1 - prefix_sum_probs_per_group + probs_eval)

        # assert prefix_sum_probs.shape[1] == tree_size

        # Get the acceptance tensor
        # torch.manual_seed(2024)
        random_tensor = torch.rand(*probs_residual_eval.shape, device=probs_residual_eval.device, dtype=probs_residual_eval.dtype)
        row_idx = torch.arange(batch_size, device=probs.device, dtype=torch.int64)
        vocab_size = probs.shape[-1]
        # assert not_debug or random_tensor.shape == probs_residual_eval.shape

        # [bs, tree_size]: [(x0), (x1, ..., x16), (x17,..., x26), (x27,..., x30), ...]
        accepted = random_tensor < probs_residual_eval
        # assert not_debug or accepted.shape[1] == bin_filter.shape[0]

        # Generic value within a group
        # 0 0 1 x     2^3 2^2 2^1 2^0      0 0 2 0           
        # 0 0 0 0  *  2^3 2^2 2^1 2^0  ->  0 0 0 0 --pref_sum--> 
        # 1 x x x     2^3 2^2 2^1 2^0      8 4 2 1           
        #
        # 0  0  2  2                -inf, -inf,    1,    1
        # 0  0  0  0  --log2(*)-->  -inf, -inf, -inf, -inf
        # 8 12 14 15                   3,  3.5,  3.8,  3.9

        # Multiple accepted tensor with a binary filter: [bs, tree_size]
        accepted_bin = accepted.float() * bin_filter
        #: [bs, tree_size]: [(x0), (x1, ..., x16), (x17,..., x26), (x27,..., x30), ...]
        prefix_sum_accepted = torch.cumsum(accepted_bin, dim=-1)
        #: [bs, tree_size]
        prefix_sum_accepted_group = prefix_sum_accepted - prefix_sum_accepted[:, parent_group_idx]
        #: [bs, tree_size]
        acc_left_shift = torch.log2_(prefix_sum_accepted_group + 1e-6)
        acc_left_shift = torch.where(torch.logical_or(acc_left_shift < -1e-6, torch.isnan(acc_left_shift)), LOG_0, acc_left_shift).type(torch.int64)
    
        # successor_tail_idx: [(16), (26, 30, 32, ..., )]

        # Accepting iteration
        device= probs_residual_eval.device
        output_tokens = torch.zeros((batch_size, tree_depth), dtype=torch.int64, device=device) # tensor_buffer: [bs, tree_depth]. Here output excludes tree root to make room for the extra tokens
        output_token_idx = torch.full((batch_size, tree_depth), -1, dtype=torch.int64, device=device) # Excludes the tree root to make room for the extra tokens
        # Note: acc_length doesn't include the resampled token, or the root token
        output_all_acc = torch.full((batch_size,), -1, dtype=torch.int64, device=device) # [bs, 2]: batch_idx -> idx_parent, acc_length
        output_resample = torch.full((batch_size,), -1, dtype=torch.int64, device=device) # [bs, 3]: batch_idx -> idx_parent, acc_length
        written_resample = torch.full((batch_size,), False, dtype=torch.bool, device=device)
        written_all_acc = torch.full((batch_size,), False, dtype=torch.bool, device=device)
        acc_lengths = torch.full((batch_size,), -1, device=device, dtype=torch.int64)

        idx_parent = torch.full((batch_size,), 0, device=accepted.device, dtype=torch.int64)
        # inital idx_cur: [16, ..., 16]. Use the group last idx to represent the group idx
        idx_cur_group = torch.full((batch_size,), first_successor_idx, device=accepted.device, dtype=torch.int64)
        # [(0), (26, 30, 32, ..., )]
        # successor_tail_idx[0] = 0  # stablizer at root; always map the leaf node back to the root, for the all-accepted sequences.
        acc_left_shift[:, 0] = 0  # stablizer at the root group; always zero shifting
        for depth in range(1, tree_depth):
            # Left shift
            shift = acc_left_shift[row_idx, idx_cur_group]
            
            # Check for the all-rejected: 
            # (shift == -inf) indicates all-rejection
            # batch_idx = torch.nonzero((shift == LOG_0) & (~written_resample))
            batch_idx = (shift == LOG_0) & (~written_resample)
            # Store them as batch_idx -> (idx_parent, acc_length)            
            # output_resample[batch_idx] = idx_parent[batch_idx]
            rubikon_cache_ops.move_entry_col_selective0(idx_parent, output_resample, batch_idx)
            # output_resample[batch_idx, 1] = depth - 1
            acc_lengths[batch_idx] = depth - 1
            written_resample[batch_idx] = True

            idx_cur = idx_cur_group - shift
            idx_cur[written_resample] = 0  # stablizer; always map the all-rejected sequences to the root.

            # Output the normal tokens
            output_token_idx[:, depth - 1] = idx_cur
            output_tokens[:, depth - 1] = draft_tokens[row_idx, idx_cur]

            # Map to next layer
            idx_parent = idx_cur
            # [(0), (26, 30, 32, ..., )]
            idx_cur_group = successor_tail_idx_stablized[idx_cur]  # stablizer in successor_tail_idx

            # Check for the all-accepted:
            # (idx_cur_group == 0 and ~resample) indicates it circles back, thus is the all-accepted request;
            batch_idx2 = (idx_cur_group == 0) & ~written_resample & ~written_all_acc
            # Store them as batch_idx2 -> (idx_parent, acc_length)
            # output_all_acc[batch_idx2] = idx_parent[batch_idx2]
            rubikon_cache_ops.move_entry_col_selective0(idx_parent, output_all_acc, batch_idx2)
            # output_all_acc[batch_idx2, 1] = depth
            acc_lengths[batch_idx2] = depth
            written_all_acc[batch_idx2] = True

        # 4. ======== Resample the all-rejected and the all-accepted ========
        # assert not_debug or torch.nonzero(output_all_acc[:, 0] != -1).view(-1).shape[0] + torch.nonzero(output_resample[:, 0] != -1).view(-1).shape[0] == batch_size
        # assert not_debug or torch.nonzero(written_resample).view(-1).shape[0] + torch.nonzero(written_all_acc).view(-1).shape[0] == batch_size
        probs_to_sample = torch.empty((batch_size, vocab_size), device=device, dtype=probs.dtype)

        # Resample:
        # output_resample: batch_idx -> idx_parent, depth
        # target_probs_residual: [bs, tree_size, vocab_size]
        batch_idx = written_resample
        # parent_idx = output_resample[batch_idx]
        # probs_to_sample[batch_idx] = probs_residual[batch_idx, parent_idx]
        rubikon_cache_ops.move_entry_col_selective1(probs_residual, output_resample, probs_to_sample, batch_idx)


        # All-accepted:
        # output_all_acc: batch_idx -> idx_parent, depth
        # target_probs: [bs, tree_size, vocab_size]
        # all_accepted requests' parent_idx may be in the last layer and out-of-bound of probs.size(1); reset them to dummy 0, i.e. the root input probs.
        batch_idx2 = written_all_acc
        # parent_idx2 = output_all_acc[batch_idx2]
        # parent_idx2[parent_idx2 >= probs.size(1)] = 0 
        # probs_to_sample[batch_idx2] = probs[batch_idx2, parent_idx2]
        rubikon_cache_ops.move_entry_col_selective1(probs, output_all_acc, probs_to_sample, batch_idx2)

        # Set the extra tokens (resampled and leaf tokens) to output_tokens
        # probs_to_sample2 = probs_to_sample.clone()
        extra_tokens = vllm_sampler._multinomial(probs_to_sample, 1)

        # row_idx = torch.arange(batch_size, device=output_tokens.device, dtype=torch.int64)
        output_tokens[row_idx, acc_lengths] = extra_tokens.view(-1)

        # Abandon the bonus-layer leaf tokens
        seq_id_bonus_bool = acc_lengths == tree_depth - 1
        rubikon_cache_ops.minus_one_selective(acc_lengths, seq_id_bonus_bool)

        # Kv_cache consolidation
        slot_source_cu = torch.full(output_token_idx.shape, -1, dtype=output_token_idx.dtype, device=output_token_idx.device)
        slot_mapping_cu = torch.full(output_token_idx.shape, -1, dtype=output_token_idx.dtype, device=output_token_idx.device)
        rubikon_cache_ops.get_slots(
            slot_source_cu,
            slot_mapping_cu,
            output_token_idx, 
            acc_lengths, 
            prefix_length, 
            block_tables,
            block_size)
        
        for kv_cache in [draft_kv_caches, target_kv_caches]:
            for key_cache, val_cache in kv_cache:
                rubikon_cache_ops.consolidate_cache_flash_layout(
                    key_cache,
                    val_cache,
                    slot_source_cu,
                    slot_mapping_cu,
                    'auto',
                )

        return output_tokens, output_token_idx, acc_lengths, probs_eval, seq_id_bonus_bool

    def speculate_decode_step(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        target_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        draft_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> SpeculativeSpeculateOutput:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         real_batch_size
         ) = self._prepare_speculate_decode(seq_group_metadata_list)
        context_lens = attn_metadata.context_lens # prompt + input_len
        slot_mapping = attn_metadata.slot_mapping
        graph_batch_size = input_tokens.shape[0]
        # 1. Run the draft model
        for i in range(self.speculate_length):
            with MarkActiveModel(ActiveModel.DRAFT):
                model = self.draft_model
                model_executable = self.draft_model
                attn_metadata.is_multi_query_mode = False
                attn_metadata.seq_len = 1
                attn_metadata.context_lens = context_lens - self.speculate_length + i
                if attn_metadata.use_cuda_graph:
                    model_executable = self.draft_graph_runners[
                        graph_batch_size]
                    # These tensors don't need to be contiguous when
                    # using cudagraph because CUDAGraphRunner will copy
                    # the inputs to be contiguous.
                    input_tokens_1d = input_tokens[:, i]
                    input_positions_1d = input_positions[:, i]
                    attn_metadata.slot_mapping = slot_mapping[:, i]
                else:
                    input_tokens_1d = input_tokens[:, i].contiguous()
                    input_positions_1d = input_positions[:, i].contiguous()
                    attn_metadata.slot_mapping = slot_mapping[:,
                                                              i].contiguous()
                hidden_states = model_executable(
                    input_ids=input_tokens_1d,
                    positions=input_positions_1d,
                    kv_caches=draft_kv_caches,
                    attn_metadata=attn_metadata,
                )
                # We always use greedy sampling to sample draft tokens.
                next_tokens = self.fast_greedy_sample(model, hidden_states)
                input_tokens[:, i + 1] = next_tokens

        # 2. Run the target model
        sampling_metadata.input_token_ids = input_tokens[:real_batch_size]
        sampling_metadata.is_multi_query_mode = True
        with MarkActiveModel(ActiveModel.TARGET):
            model = self.model
            model_executable = self.model
            attn_metadata.is_multi_query_mode = True
            attn_metadata.seq_len = self.speculate_length + 1
            input_tokens_1d = input_tokens.view(-1)
            input_positions_1d = input_positions.view(-1)
            attn_metadata.slot_mapping = slot_mapping.view(-1)
            attn_metadata.context_lens = context_lens
            if attn_metadata.use_cuda_graph:
                graph_runners = self.graph_runners
                # For speculative decoding, cudagraph is only used in the evaluation stage
                # for target model.
                model_executable = graph_runners[graph_batch_size]
            # Execute the model.
            hidden_states = model_executable(
                input_ids=input_tokens_1d,
                positions=input_positions_1d,
                kv_caches=target_kv_caches,
                attn_metadata=attn_metadata,
            )
            bs, num_tokens = input_tokens.shape
            hidden_states = hidden_states.view(bs, num_tokens, -1)[:real_batch_size]
            # Compute the logits.
            logits = model.compute_logits(hidden_states, sampling_metadata)
            # Only perform sampling in the driver worker.
            if not sampling_metadata.perform_sampling:
                return None
            # Sample the next token.
            output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        # 3. Collect output
        speculate_outputs: SpeculativeSpeculateOutput = []
        for seq_group_output in output:
            # 1. Handling the case when multiple tokens can be generated.
            # Here we drop the last sampled token when all draft tokens were accepted.
            # When all draft tokens were accepted, the last 2 generated tokens (the last
            # accepted draft token plus the token sampled from its logits) will miss
            # their kv caches in the draft model and requires multi-query attention.
            # Mixing single query and multi-query attention is currently not supported.
            seq_id = seq_group_output.parent_seq_id
            num_accepted = seq_group_output.num_accepted_tokens
            max_num_generated_tokens = min(num_accepted + 1,
                                           sampling_metadata.speculate_length)
            output_tokens = seq_group_output.output_tokens[:
                                                           max_num_generated_tokens]
            # print0(f'token_list {output_tokens}')
            output_token_logprobs = seq_group_output.logprobs_list[:
                                                                   max_num_generated_tokens]
            speculate_outputs.append(
                SpeculativeSpeculateSequenceGroupOutput(seq_id,
                                                        output_tokens,
                                                        output_token_logprobs,
                                                        num_accepted,
                                                        prompt_logprobs=None))
        return speculate_outputs

    @torch.inference_mode()
    def speculate_execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        is_prompt: bool,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        draft_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        step=0
    ) -> SpeculativeSpeculateOutput:
        if is_prompt:
            return self.speculate_prefill_step(seq_group_metadata_list,
                                               kv_caches, draft_kv_caches)
        if self.speculate_tree:
            return self.speculate_decode_step_2d(seq_group_metadata_list, kv_caches,
                                            draft_kv_caches, step=step)
        else:
            return self.speculate_decode_step(seq_group_metadata_list, kv_caches,
                                            draft_kv_caches)

    def __del__(self) -> None:
        # Delete the CUDA graphs before deleting the CuPy NCCL communicator.
        # NOTE(woosuk): This is necessary because otherwise deadlocks can
        # happen.
        # FIXME(woosuk): This is a bit hacky. Find a more robust solution.
        self.graph_runners.clear()
        self.draft_graph_runners.clear()
        self.cupy_nccl_backend = None

def print0(s, switch=True):
    if torch.cuda.current_device() == 0 and switch:
        print(s)
