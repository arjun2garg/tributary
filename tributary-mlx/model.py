import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

class PartialModel:
    def __init__(self, model_path: str, start_layer: int = 0, end_layer: int | None = None):
        self.model, self.tokenizer = load(model_path, lazy=True)
        self.num_layers = len(self.model.model.layers)
        # Activations cross the wire as float16, but the model computes in its own
        # dtype (bfloat16 for Qwen3). Feeding a float16 activation to a bfloat16
        # quantized_matmul drops it onto an unbatched per-row path — verify(T) then
        # costs ~T× a single forward instead of batching. Cast incoming activations
        # back to the compute dtype at the model boundary to keep the batched gemm.
        self.dtype = self.model.model.norm.weight.dtype
        self.start_layer = start_layer
        self.end_layer = end_layer if end_layer is not None else self.num_layers
        self.cache = None
        self._spec_x: list[int] | None = None
        self._spec_q: mx.array | None = None
        self._spec_key: mx.array | None = None
        self._last_probs: mx.array | None = None

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
        x = hidden_states.astype(self.dtype)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1]).astype(x.dtype)
        for i in range(self.start_layer, self.end_layer):
            x = self.model.model.layers[i](x, mask=mask, cache=self.cache[i - self.start_layer])
        mx.eval(x)
        return x
    
    def decode_step(self, hidden_states: mx.array) -> mx.array:
        assert self.cache is not None
        x = hidden_states.astype(self.dtype)
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

    def draft_sample(self, cur: int, k: int, temperature: float, seed: int) -> list[int]:
        assert self.cache is not None
        key = mx.random.key(2 * seed)
        xs: list[int] = []
        q_rows: list[mx.array] = []
        tok = cur
        for _ in range(k):
            logits = self.decode_logits(self.decode_step(self.embed([tok])))[0, -1, :]
            scaled = logits.astype(mx.float32) / temperature
            probs = mx.softmax(scaled)
            key, sub = mx.random.split(key)
            tok = int(mx.random.categorical(scaled[None], key=sub).item())
            xs.append(tok)
            q_rows.append(probs)
        mx.eval(self.decode_step(self.embed([xs[-1]])))
        self._spec_x = xs
        self._spec_q = mx.stack(q_rows, axis=0)
        return xs

    def _probs(self, hidden_states: mx.array, temperature: float) -> mx.array:
        """Temperature-scaled softmax over the vocab in float32; drops the batch dim -> [T, vocab]."""
        return mx.softmax(self.decode_logits(hidden_states).astype(mx.float32) / temperature, axis=-1)[0]

    @staticmethod
    def _sample_from_probs(probs: mx.array, key: mx.array) -> int:
        """Sample one token id from a 1D prob row, guarding zeros before the log."""
        logits = mx.where(probs > 0.0, mx.log(probs), -1e30)
        return int(mx.random.categorical(logits[None], key=key).item())

    @staticmethod
    def _residual(p: mx.array, q: mx.array) -> mx.array:
        """Normalized positive residual (p - q)+, falling back to p if it vanishes."""
        resid = mx.maximum(p - q, 0.0)
        total = float(resid.sum().item())
        return p if total <= 0.0 else (resid / total)

    def spec_accept(self, p_probs: mx.array, seed: int) -> tuple[int, int]:
        assert self._spec_x is not None and self._spec_q is not None
        x, q = self._spec_x, self._spec_q
        k = len(x)
        key = mx.random.key(2 * seed + 1)
        a = 0
        final: int | None = None
        for j in range(k):
            px = float(p_probs[j, x[j]].item())
            qx = float(q[j, x[j]].item())
            key, sub = mx.random.split(key)
            r = float(mx.random.uniform(key=sub).item())
            ratio = 1.0 if qx <= 0.0 else min(1.0, px / qx)
            if r < ratio:
                a += 1
                continue
            key, sub = mx.random.split(key)
            final = self._sample_from_probs(self._residual(p_probs[j], q[j]), sub)
            break
        if final is None:
            key, sub = mx.random.split(key)
            final = self._sample_from_probs(p_probs[k], sub)
        self._spec_x = None
        self._spec_q = None
        return a, final

    def verify_scalars(self, hidden_states: mx.array, x: list[int], temperature: float) -> list[float]:
        p = self._probs(hidden_states, temperature)
        self._last_probs = p
        return [float(p[j, x[j]].item()) for j in range(len(x))]

    def logits_at(self, pos: int) -> mx.array:
        assert self._last_probs is not None
        return self._last_probs[pos][None]  # [1, vocab]

    def accept_scalars(self, px: list[float], seed: int) -> tuple[int, int]:
        assert self._spec_x is not None and self._spec_q is not None
        x, q = self._spec_x, self._spec_q
        k = len(x)
        key = mx.random.key(2 * seed + 1)
        a = 0
        for j in range(k):
            qx = float(q[j, x[j]].item())
            key, sub = mx.random.split(key)
            r = float(mx.random.uniform(key=sub).item())
            ratio = 1.0 if qx <= 0.0 else min(1.0, px[j] / qx)
            if r < ratio:
                a += 1
                continue
            self._spec_key = key
            return a, j
        self._spec_key = key
        return a, k

    def resample_at(self, p_row: mx.array, pos: int) -> int:
        assert self._spec_key is not None and self._spec_x is not None and self._spec_q is not None
        key = self._spec_key
        q = self._spec_q
        k = len(self._spec_x)
        key, sub = mx.random.split(key)
        src = self._residual(p_row[0], q[pos]) if pos < k else p_row[0]
        final = self._sample_from_probs(src, sub)
        self._spec_x = None
        self._spec_q = None
        self._spec_key = None
        return final

    def verify_probs(self, hidden_states: mx.array, temperature: float) -> mx.array:
        return self._probs(hidden_states, temperature)

    def verify_accept(self, hidden_states: mx.array, x: list[int], temperature: float, seed: int) -> tuple[int, int]:
        p = self._probs(hidden_states, temperature)
        k = len(x)
        vocab = p.shape[-1]
        key = mx.random.key(seed)
        a = 0
        final: int | None = None
        for j in range(k):
            key, sub = mx.random.split(key)
            r = float(mx.random.uniform(key=sub).item())
            if r < float(p[j, x[j]].item()):
                a += 1
                continue
            resid = mx.where(mx.arange(vocab) == x[j], 0.0, p[j])
            resid = resid / resid.sum()
            key, sub = mx.random.split(key)
            final = self._sample_from_probs(resid, sub)
            break
        if final is None:
            key, sub = mx.random.split(key)
            final = self._sample_from_probs(p[k], sub)
        return a, final

    def decode_logits(self, hidden_states: mx.array) -> mx.array:
        x = self.model.model.norm(hidden_states)
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(x)
        return self.model.lm_head(x)
    
    def sample_token(self, logits: mx.array, temperature: float = 0.0, seed: int | None = None) -> int:
        last_logits = logits[0, -1, :]
        if temperature == 0.0:
            return mx.argmax(last_logits).item()
        scaled = last_logits / temperature
        if seed is None:
            return mx.random.categorical(scaled).item()
        return mx.random.categorical(scaled[None], key=mx.random.key(seed)).item()
