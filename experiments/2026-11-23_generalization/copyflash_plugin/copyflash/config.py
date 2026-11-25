"""Config of the 'copyflash' speculative method: a DFlash speculator whose verify width is the copy width.

speculative_config = {
  "method": "copyflash", "model": "<DFlash drafter>", "num_speculative_tokens": 31,   # verify width (copy length)
  "speculator_options": {"dflash_k": 15,      # DFlash block - 1 (the drafter's own width)
                         "router_min": 8,     # copy when the current suffix (>= this many tokens) occurs earlier
                         "max_n": 24,         # longest suffix considered when locating the copy source
                         "mode": "router"}    # "router" | "dflash_only" (DFlash padded to the verify width)
}
"""
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig


class CopyFlashSpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = 'copyflash.speculator.CopyFlashSpeculator'
    supported_options = frozenset({'speculator_options'})
    # the copy branch scans the request's full token history on the GPU (as the built-in ngram speculator does)
    prefer_dense_token_history = True

    @classmethod
    def matches(cls, spec) -> bool:
        return False  # selected only by method name

    @property
    def dflash_k(self) -> int:
        return int((self.spec_config.speculator_options or {}).get('dflash_k', 15))

    @property
    def num_query_tokens(self) -> int:
        # the drafter keeps its trained block: dflash_k drafts (+ the anchor in the fill-in layout),
        # independent of the (wider) verify width num_speculative_tokens
        return self.dflash_k + (0 if self.sample_from_anchor else 1)
