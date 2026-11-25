"""Copy + DFlash router as a vLLM V2 speculator.

Every step, for every request:
  1. DFlash runs exactly as stock (one drafter pass, dflash_k drafts). This also stores the drafter's context K/V
     for every token the target just verified, so the drafter stays in sync whichever branch proposed them.
  2. The built-in GPU n-gram kernels look for the current suffix (router_min..max_n tokens, longest then most
     recent) earlier in the prompt + output, and read up to K = num_speculative_tokens tokens after it.
  3. Router: a match of at least router_min tokens -> the copy draft (K tokens), else the DFlash draft padded to K
     (padding is ordinary rejectable drafts; it only costs verify width).
All on the GPU: no host sync, CUDA graphs and async scheduling unchanged. Verification is the stock greedy
rejection sampler, so the output is the target's own greedy output.
"""
import copy

import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import _ngram_finalize_kernel, _ngram_scan_kernel

logger = init_logger('vllm.copyflash')  # vLLM only configures handlers for 'vllm.*'


class CopyFlashSpeculator(DFlashSpeculator):
    _speculator_name = 'CopyFlash'

    def __init__(self, vllm_config, device, req_states):
        spec = vllm_config.speculative_config
        opts = spec.speculator_options or {}
        self.copy_k = int(spec.num_speculative_tokens)
        dflash_k = int(opts.get('dflash_k', 15))
        self.router_min = int(opts.get('router_min', 8))
        self.max_n = max(int(opts.get('max_n', 24)), self.router_min)
        self.mode = opts.get('mode', 'router')
        assert self.mode in ('router', 'dflash_only'), self.mode
        assert self.copy_k >= dflash_k >= 1
        # The DFlash internals (drafter query block, sampling buffers, draft_tokens) are built for dflash_k;
        # only this wrapper's output and the verify width use copy_k.
        inner_spec = copy.copy(spec)
        object.__setattr__(inner_spec, 'num_speculative_tokens', dflash_k)
        inner = copy.copy(vllm_config)
        object.__setattr__(inner, 'speculative_config', inner_spec)
        super().__init__(inner, device)
        assert self.num_speculative_steps == dflash_k

        self.req_states = req_states
        L = self.max_model_len
        self.block_l = 256 if L >= 1024 else 128 if L >= 256 else 64 if L >= 64 else max(16, triton.next_power_of_2(max(L, 1)))
        self.n_blocks = triton.cdiv(L, self.block_l)
        self.scratch = torch.zeros((self.max_num_reqs, self.n_blocks), dtype=torch.int64, device=device)
        self.copy_drafts = torch.zeros((self.max_num_reqs, self.copy_k), dtype=torch.int64, device=device)
        self.out = torch.zeros((self.max_num_reqs, self.copy_k), dtype=torch.int64, device=device)
        self.n_steps = torch.zeros((), dtype=torch.int64, device=device)
        self.n_copy = torch.zeros((), dtype=torch.int64, device=device)
        self.n_finished = 0
        logger.info('CopyFlash: verify width %d, DFlash width %d, router_min %d, max_n %d, mode %s',
                    self.copy_k, dflash_k, self.router_min, self.max_n, self.mode)

    def _num_graph_context_tokens(self, num_reqs: int) -> int:
        # the target verifies copy_k + 1 tokens per request, so the FULL graph stores that many context rows
        return min(num_reqs * (self.copy_k + 1), self.max_num_tokens)

    @torch.inference_mode()
    def propose(self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds,
                dp_sync_state=None, dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False, num_speculative_tokens=None):
        inner_k = None if num_speculative_tokens is None else min(num_speculative_tokens, self.num_speculative_steps)
        d = super().propose(input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
                            num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds,
                            dp_sync_state=dp_sync_state, dummy_run=dummy_run,
                            skip_attn_for_dummy_run=skip_attn_for_dummy_run, mm_inputs=mm_inputs,
                            is_profile=is_profile, num_speculative_tokens=inner_k)
        n = d.shape[0]
        k = self.num_speculative_steps
        out = self.out[:n]
        out[:, :k] = d
        if self.copy_k > k:
            out[:, k:] = d[:, -1:]  # padding: verified like any draft token, accepted only if the target agrees
        if dummy_run or self.mode == 'dflash_only' or num_speculative_tokens == 0:
            return out

        num_reqs = input_batch.num_reqs
        token_ids = self.req_states.all_token_ids.gpu
        idx_mapping = input_batch.idx_mapping
        _ngram_scan_kernel[(num_reqs, self.n_blocks)](
            token_ids, token_ids.stride(0), idx_mapping, self.req_states.total_len.gpu, num_sampled,
            self.scratch, self.scratch.stride(0), self.max_model_len, self.router_min, self.max_n,
            max(1, triton.next_power_of_2(self.max_n)), self.block_l, num_warps=4, num_stages=2)
        _ngram_finalize_kernel[(num_reqs,)](
            token_ids, token_ids.stride(0), idx_mapping, self.req_states.total_len.gpu, num_sampled,
            last_sampled.view(-1), self.scratch, self.scratch.stride(0), self.copy_drafts, self.max_model_len,
            self.n_blocks, self.copy_k, max(1, triton.next_power_of_2(self.copy_k)),
            max(1, triton.next_power_of_2(self.n_blocks)), num_warps=2, num_stages=1)
        use_copy = self.scratch[:num_reqs].amax(dim=1) > 0  # a suffix of >= router_min tokens occurs earlier
        out[:num_reqs] = torch.where(use_copy[:, None], self.copy_drafts[:num_reqs], out[:num_reqs])
        self.n_steps += num_reqs
        self.n_copy += use_copy.sum()
        return out

    def on_requests_finished(self, finished_req_ids) -> None:
        super().on_requests_finished(finished_req_ids)
        self.n_finished += len(finished_req_ids)
        # one host sync per finished request: negligible, and it gives the copy-step share in the server log
        logger.info('COPYFLASH finished=%d steps=%d copy_steps=%d', self.n_finished, int(self.n_steps),
                    int(self.n_copy))
