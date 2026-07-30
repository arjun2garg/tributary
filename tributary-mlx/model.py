import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

class PartialModel:
    def __init__(self, model_path: str, start_layer: int = 0, end_layer: int | None = None):
        self.model, self.tokenizer = load(model_path)
        self.num_layers = len(self.model.model.layers)
        self.start_layer = start_layer
        self.end_layer = end_layer if end_layer is not None else self.num_layers
        self.cache = None

        if not (0 <= self.start_layer < self.end_layer <= self.num_layers):
            raise ValueError(
                f"invalid layer range {self.start_layer}..{self.end_layer} "
                f"(model has {self.num_layers} layers)"
            )
    
    @property
    def is_first(self) -> bool:
        return self.start_layer == 0
    
    @property
    def is_last(self) -> bool:
        return self.end_layer == self.num_layers
    
    def reset_cache(self):
        self.cache = None

    def embed(self, token_ids: list[int]) -> mx.array:
        ids = mx.array(token_ids)[None]
        return self.model.model.embed_tokens(ids)

    def prefill(self, hidden_states: mx.array) -> mx.array:
        self.cache = make_prompt_cache(self.model)[self.start_layer:self.end_layer]
        x = hidden_states
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(x.dtype)
        for i in range(self.start_layer, self.end_layer):
            x = self.model.model.layers[i](x, mask=mask, cache=self.cache[i - self.start_layer])
        mx.eval(x)
        return x
    
    def decode_step(self, hidden_states: mx.array) -> mx.array:
        assert self.cache is not None
        x = hidden_states
        for i in range(self.start_layer, self.end_layer):
            x = self.model.model.layers[i](x, mask=None, cache=self.cache[i - self.start_layer])
        return x
    
    def decode_logits(self, hidden_states: mx.array) -> mx.array:
        x = self.model.model.norm(hidden_states)
        return self.model.model.embed_tokens.as_linear(x)
    
    def sample_token(self, logits: mx.array, temperature: float = 0.0) -> int:
        last_logits = logits[0, -1, :]
        if temperature == 0.0:
            return mx.argmax(last_logits).item()
        else:
            probs = mx.softmax(last_logits / temperature)
            return mx.random.categorical(mx.log(probs)).item()
