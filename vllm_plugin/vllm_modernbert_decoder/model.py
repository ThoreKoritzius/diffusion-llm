"""vLLM implementation of ModernBertDecoderForCausalLM (Ettin decoders), mirroring transformers'
models/modernbert_decoder/modeling_modernbert_decoder.py:

- token embedding + LayerNorm
- pre-norm decoder layers (no attention norm in layer 0): separate q/k/v projections, NeoX-style RoPE, causal
  attention that is global or sliding-window per `config.layer_types`, GeGLU MLP (Wi -> gelu(x) * gate -> Wo)
- final LayerNorm, prediction head (dense -> gelu -> LayerNorm), output projection tied to the token embedding with
  its own bias (`decoder.bias`)

Tensor parallelism is not supported (the fused Wi projection is not split into input/gate shards).
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import GeluAndMul, get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import extract_layer_index, maybe_prefix
from vllm.sequence import IntermediateTensors


def _rope_theta(config, sliding: bool) -> float:
    theta = getattr(config, "local_rope_theta" if sliding else "global_rope_theta", None)
    if theta:
        return float(theta)
    params = getattr(config, "rope_parameters", None) or {}
    per_type = params.get("sliding_attention" if sliding else "full_attention")
    if isinstance(per_type, dict):
        return float(per_type.get("rope_theta", 10000.0))
    return float(params.get("rope_theta", 10000.0))


def _layer_norm(config) -> nn.LayerNorm:
    return nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)


class ModernBertDecoderAttention(nn.Module):
    def __init__(self, config, cache_config: CacheConfig | None, quant_config: QuantizationConfig | None,
                 prefix: str):
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.qkv = QKVParallelLinear(config.hidden_size, self.head_dim, self.num_heads, self.num_heads,
                                     bias=config.attention_bias, quant_config=quant_config, prefix=f"{prefix}.qkv")
        self.Wo = RowParallelLinear(self.q_size, config.hidden_size, bias=config.attention_bias,
                                    quant_config=quant_config, prefix=f"{prefix}.Wo")
        self.rotary_emb = get_rope(self.head_dim, max_position=config.max_position_embeddings, is_neox_style=True,
                                   rope_parameters={"rope_type": "default", "rope_theta": _rope_theta(config, sliding)})
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim**-0.5, num_kv_heads=self.num_heads,
                              cache_config=cache_config, quant_config=quant_config,
                              per_layer_sliding_window=config.sliding_window if sliding else None,
                              prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.q_size, self.q_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        out, _ = self.Wo(self.attn(q, k, v))
        return out


class ModernBertDecoderMLP(nn.Module):
    def __init__(self, config, quant_config: QuantizationConfig | None, prefix: str):
        super().__init__()
        self.Wi = ColumnParallelLinear(config.hidden_size, 2 * config.intermediate_size, bias=config.mlp_bias,
                                       quant_config=quant_config, prefix=f"{prefix}.Wi")
        if config.hidden_activation not in ("gelu", "gelu_pytorch_tanh", "gelu_new"):
            raise ValueError(f"unsupported activation {config.hidden_activation}")
        self.act = GeluAndMul(approximate="none" if config.hidden_activation == "gelu" else "tanh")
        self.Wo = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias,
                                    quant_config=quant_config, prefix=f"{prefix}.Wo")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.Wi(x)
        out, _ = self.Wo(self.act(h))
        return out


class ModernBertDecoderLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str):
        super().__init__()
        self.attn_norm = nn.Identity() if extract_layer_index(prefix) == 0 else _layer_norm(config)
        self.attn = ModernBertDecoderAttention(config, cache_config, quant_config, prefix=f"{prefix}.attn")
        self.mlp_norm = _layer_norm(config)
        self.mlp = ModernBertDecoderMLP(config, quant_config, prefix=f"{prefix}.mlp")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(positions, self.attn_norm(hidden_states))
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class ModernBertDecoderEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.norm = _layer_norm(config)


@support_torch_compile
class ModernBertDecoderModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embeddings = ModernBertDecoderEmbeddings(config)
        self.layers = nn.ModuleList([
            ModernBertDecoderLayer(config, vllm_config.cache_config, vllm_config.quant_config,
                                   prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])
        self.final_norm = _layer_norm(config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings.tok_embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        hidden_states = self.embeddings.norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.final_norm(hidden_states)


class ModernBertDecoderPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.classifier_bias)
        self.act = get_act_fn(config.classifier_activation)
        self.norm = _layer_norm(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


class ModernBertDecoderForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("ModernBertDecoderForCausalLM: tensor parallelism is not supported")
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = ModernBertDecoderModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ModernBertDecoderPredictionHead(config)
        emb = self.model.embeddings.tok_embeddings
        # output projection is tied to the token embedding (padded vocab rows); its bias is a separate parameter
        self.decoder_bias = nn.Parameter(torch.zeros(emb.num_embeddings_per_partition), requires_grad=False) \
            if getattr(config, "decoder_bias", True) else None
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.model.embeddings.tok_embeddings, self.lm_head(hidden_states),
                                     self.decoder_bias)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if name == "decoder.weight":  # tied to model.embeddings.tok_embeddings
                continue
            if name == "decoder.bias":
                if self.decoder_bias is not None:
                    self.decoder_bias.data.zero_()
                    self.decoder_bias.data[: weight.shape[0]].copy_(weight)
                    loaded.add("decoder_bias")
                continue
            for hf_name, shard in ((".attn.q_proj.", "q"), (".attn.k_proj.", "k"), (".attn.v_proj.", "v")):
                if hf_name in name:
                    target = name.replace(hf_name, ".attn.qkv.")
                    param = params[target]
                    param.weight_loader(param, weight, shard)
                    loaded.add(target)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
                loaded.add(name)
        return loaded
