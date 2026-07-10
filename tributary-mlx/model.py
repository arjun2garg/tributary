import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

class PartialModel:
    def __init__(self, model_path: str, start_layer: int, end_layer: int):
        self.model, self.tokenizer = load(model_path)
        self.start_layer = start_layer
        self.end_layer = end_layer

    def embed(self, prompt: str) -> mx.array:
        tokens = self.tokenizer.encode(prompt)
        token_ids = mx.array(tokens)[None]
        return self.model.model.embed_tokens(token_ids)
    
    def forward_partial(self, hidden_states: mx.array) -> mx.array:
        x = hidden_states
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(x.dtype)
        for i in range(self.start_layer, self.end_layer):
            x = self.model.model.layers[i](x, mask=mask)        
        return x
    
    def decode(self, hidden_states: mx.array) -> int:
        x = self.model.model.norm(hidden_states)
        logits = self.model.model.embed_tokens.as_linear(x)
        return mx.argmax(logits[0, -1, :]).item() # greedy sampling for now