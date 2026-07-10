import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

class PartialModel:
    def __init__(self, model_path: str, start_layer: int, end_layer: int):
        self.model, self.tokenizer = load(model_path)
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.cache = None
    
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
