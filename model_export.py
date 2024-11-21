import coremltools as ct
import numpy as np
import torch
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from typing import Any, Optional, Sequence
from transformers.cache_utils import Cache

model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

batch_size, context_size = 1, 8192

# Create KV cache
class SliceUpdateKeyValueCache(Cache):
    """Helper class for in-place slice updating key/value caches."""

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create key/value cache of shape:
        (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k: torch.Tensor = torch.zeros(shape, dtype=dtype)
        self.v: torch.Tensor = torch.zeros(shape, dtype=dtype)

    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key / value cache tensors for slice [begin, end).
        Return slice of key / value cache tensors from [0, end)."""
        position = cache_kwargs.get("cache_position", None)
        assert position is not None, "cache_position required to update cache."
        begin, end = self.past_seen_tokens, self.past_seen_tokens + position.shape[-1]
        self.k[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_state = self.k[layer_idx, :, :, :end, :]
        v_state = self.v[layer_idx, :, :, :end, :]
        return k_state, v_state

    def get_seq_length(self, _: int = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens

class KvCacheStateLlamaForCausalLM(torch.nn.Module):
    """Model wrapper to swap cache implementation and register as buffers."""

    def __init__(
        self,
        model_path: str, *,
        batch_size: int = 1, context_size: int = 4096
    ) -> None:
        super().__init__()
        print("Retrieving base model\n")
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        config: LlamaConfig = self.model.config
        self.kv_cache_shape: tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            context_size,
            config.hidden_size // config.num_attention_heads,
        )
        # Register KV cache buffers to be recognized as Core ML states
        print("Registering cache\n")
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k)
        self.register_buffer("valueCache", self.kv_cache.v)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits

# Initialize and trace PyTorch model using KV Cache
print("Starting\n")
loaded_model: torch.nn.Module = KvCacheStateLlamaForCausalLM(
    model_id, batch_size=batch_size, context_size=context_size
).eval()
example_inputs: tuple[torch.Tensor, ...] = (
    torch.zeros((1, 2), dtype=torch.int32),
    torch.zeros((1, 1, 2, 5), dtype=torch.float32)
)
print("Tracing\n")
traced_model: torch.jit.ScriptModule = torch.jit.trace(
    loaded_model.eval(), example_inputs=example_inputs
)

# Convert to Core ML
query_size = ct.RangeDim(lower_bound=1, upper_bound=context_size, default=1)
final_step = ct.RangeDim(lower_bound=1, upper_bound=context_size, default=1)
inputs: list[ct.TensorType] = [
    ct.TensorType(shape=(batch_size, query_size), dtype=np.int32, name="inputIds"),
    ct.TensorType(
        shape=(batch_size, 1, query_size, final_step),
        dtype=np.float16,
        name="causalMask",
    ),
]
states: list[ct.StateType] = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=loaded_model.kv_cache_shape, dtype=np.float16),
        name="keyCache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=loaded_model.kv_cache_shape, dtype=np.float16),
        name="valueCache",
    ),
]
outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
print("Converting to Core ML\n")
mlmodel: ct.models.MLModel = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    states=states,
    minimum_deployment_target=ct.target.macOS15,
    skip_model_load=True,
)

# Quantize
op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int4",
    granularity="per_block",
    block_size=32,
)
config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
print("Quantizing\n")
mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(
    mlmodel, config=config
)
print("Init state\n")
kv_cache = mlmodel_int4.make_state()
print("Predict\n")
logits = mlmodel_int4.predict(data=inputs, state=kv_cache)["logits"]
