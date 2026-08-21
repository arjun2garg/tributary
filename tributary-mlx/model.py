import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

class PartialModel:
    def __init__(self, model_path: str, start_layer: int = 0, end_layer: int | None = None):
        self.model, self.tokenizer = load(model_path, lazy=True)
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
        mask = create_attention_mask(x, self.cache[0])
        for i in range(self.start_layer, self.end_layer):
            x = self.model.model.layers[i](x, mask=mask, cache=self.cache[i - self.start_layer])
        return x

    def trim(self, n: int) -> None:
        if self.cache is not None and n > 0:
            trim_prompt_cache(self.cache, n)

    def draft_generate(self, cur: int, k: int) -> list[int]:
        assert self.cache is not None
        out: list[int] = []
        tok = cur
        for _ in range(k):
            h = self.embed([tok])
            h = self.decode_step(h)
            logits = self.decode_logits(h)
            tok = int(mx.argmax(logits[0, -1, :]).item())
            out.append(tok)
        h = self.decode_step(self.embed([out[-1]]))
        mx.eval(h)
        return out

    def greedy_all(self, hidden_states: mx.array) -> list[int]:
        logits = self.decode_logits(hidden_states)
        toks = mx.argmax(logits, axis=-1)  # [1, T]
        return [int(t) for t in toks[0].tolist()]

    def decode_logits(self, hidden_states: mx.array) -> mx.array:
        x = self.model.model.norm(hidden_states)
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(x)
        return self.model.lm_head(x)
    
    def sample_token(self, logits: mx.array, temperature: float = 0.0) -> int:
        last_logits = logits[0, -1, :]
        if temperature == 0.0:
            return mx.argmax(last_logits).item()
        else:
            probs = mx.softmax(last_logits / temperature)
            return mx.random.categorical(mx.log(probs)).item()
