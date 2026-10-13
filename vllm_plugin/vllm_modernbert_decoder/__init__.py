"""vLLM plugin: registers ModernBertDecoderForCausalLM (e.g. jhu-clsp/ettin-decoder-*). Loaded automatically by vLLM
through the `vllm.general_plugins` entry point once this package is installed."""


def register():
    from vllm import ModelRegistry

    if "ModernBertDecoderForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("ModernBertDecoderForCausalLM",
                                     "vllm_modernbert_decoder.model:ModernBertDecoderForCausalLM")
