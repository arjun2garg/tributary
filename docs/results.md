# Baseline

Model: Llama-3.2-3B-Instruct-4bit  
Hardware: 2023 Macbook Air, M2 Chip, 8GB RAM
Date: July 9, 2026

## Single Node Performance
- Tokens/sec: 2.4 tok/s

## Notes
- Verified that split partial pass is identical to full pass
- KV cache not implemented

# KV Cache Baseline

Model: Llama-3.2-3B-Instruct-4bit  
Hardware: 2023 Macbook Air, M2 Chip, 8GB RAM
Date: July 10, 2026

## Single Node Performance

| Prompt length | Naive (tok/s) | Cached (tok/s) | Speedup |
|--------------:|--------------:|---------------:|--------:|
|            10 |           6.8 |           39.0 |    5.7x |
|            50 |           3.3 |           39.1 |   11.7x |
|           100 |           2.1 |           27.6 |   13.1x |
|           250 |           0.6 |           28.4 |   44.9x |

## Notes
- Verified cached results are identical to naive implementation

# Rust Generation Loop

Model: Llama-3.2-3B-Instruct-4bit  
Hardware: 2023 Macbook Air, M2 Chip, 8GB RAM
Date: July 10, 2026

Control inversion: Rust drives the generation loop via 5 localhost HTTP calls per token
(`/detokenize`, `/embed`, `/forward?mode=decode`, `/logits?last_only=true`, `/sample`)
against a single full-range MLX server. The Δ vs the Python-side loop is the per-token
IPC (localhost HTTP) cost — measured single-node so later two-node numbers can be
decomposed into IPC vs network overhead.

## Single Node Performance

100 generated tokens per run, greedy (temperature 0). Python loop = `/generate` SSE path
(`--legacy`), re-measured in the same session on the same prompts.

| Prompt tokens | Python loop (tok/s) | Rust loop (tok/s) | Δ per token |
|--------------:|--------------------:|------------------:|------------:|
|            11 |                38.6 |              34.6 |     ~3.0 ms |
|            54 |                37.9 |              34.4 |     ~2.7 ms |
|           108 |                38.1 |              34.3 |     ~2.9 ms |
|           270 |                38.7 |              32.6 |     ~4.8 ms |

## Notes
- Greedy output is byte-identical between the two paths for all four prompts, plus an
  EOS-terminating prompt (stops before max_tokens) — control inversion is lossless
- IPC cost ≈ 3–5 ms/token across 5 HTTP calls (~0.6–1 ms per localhost round trip);
  well below the threshold where fused endpoints would be needed
- Logits response + sample request dominate the per-token wire traffic (~512 KB/token,
  vocab 128,256 × fp16) — negligible on loopback, will matter over WiFi

# Two-Process Split Pipeline

Model: Llama-3.2-3B-Instruct-4bit  
Hardware: 2023 Macbook Air, M2 Chip, 8GB RAM
Date: July 14, 2026

Model split across two local server processes — A: layers 0–14 (embeds), B: layers
14–28 (final norm + lm_head) — chained by the Rust loop over loopback: embed + forward
on A → forward + logits on B → sample. Each process holds its own KV cache.

## Single Node Performance (two processes)

| Run | Tokens | Split (tok/s) | Single-process Rust loop (tok/s) |
|---|---:|---:|---:|
| ~250-token prompt, 100 generated | 100 | 25.6 | ~33 |
| Short prompt, EOS at 78 | 78 | 22.8 | 33.0 |

## Notes
- Greedy output byte-identical to the single-process Rust loop (same prompt, diffed
  against the saved transcript) — per-process KV caches and the layer-14 activation
  handoff are correct
- Slowdown is ~10–15 ms/token, far more than the +1 HTTP call/token (~1 ms) predicts —
  both MLX processes contend for the same GPU/memory bandwidth, and each loads the full
  ~1.8 GB weights (compute is sliced, loading is not), so ~3.6 GB of an 8 GB machine
  is weights
- First generation after server start is much slower (ttft 8–23 s observed) — weight
  page-in / warmup, not steady-state

# Two-Node Pipeline (Two Physical Machines)

Model: Llama-3.2-3B-Instruct-4bit  
Coordinator: 2023 MacBook Air, M2, 8GB RAM (embed + layers 0–14 + sampling)  
Worker: second Mac (layers 14–28 + final norm + lm_head), listening on `0.0.0.0:9000`  
Date: August 17, 2026

Coordinator drives the Rust generation loop; per decoded token it runs its local layer
range, ships the hidden activation to the worker over a persistent length-prefixed TCP
frame, and the worker returns logits. One worker instance serves both transports
(reachable on its Thunderbolt IP and its WiFi IP). 200 generated tokens/run, greedy
(temperature 0), one worker warm-up discarded.

## Throughput

| Prompt tokens | Thunderbolt (tok/s) | WiFi (tok/s) | TB ttft (s) | WiFi ttft (s) |
|--------------:|--------------------:|-------------:|------------:|--------------:|
|            12 |                20.9 |         11.2 |        0.13 |          0.24 |
|            59 |                21.2 |         11.3 |        0.25 |          0.38 |
|           110 |                21.1 |         11.3 |        0.35 |          0.52 |
|           256 |                21.2 |         11.4 |        0.72 |          0.98 |

Throughput is flat across prompt length (KV cache working as expected); the transport is
the only lever — Thunderbolt is ~1.9× WiFi.

## Per-token decode latency (median, ms)

`round-trip` = activation send + worker compute + logits return (one TCP exchange);
`local` = coordinator embed + layers 0–14; `serialize`/`deserialize` are <0.01 ms and omitted.

| Transport | local | network | worker | round-trip | sample |
|-----------|------:|--------:|-------:|-----------:|-------:|
| Thunderbolt |  19.4 |    0.77 |   22.6 |       23.4 |    2.0 |
| WiFi        |  21.0 |   29.8  |   22.1 |       52.5 |    2.6 |

## Wire sizes (measured)

| Transfer | Size |
|---|---|
| Decode activation (1 token, out) | 6,144 B (~6 KB) |
| Logits (1 position, back) | 256,512 B (~256 KB) |
| Prefill activation, 256-token prompt (out) | 1,572,864 B (~1.5 MB) |

Matches the step-2 design estimates exactly (hidden 3072 × fp16; vocab 128,256 × fp16).

## Notes
- Greedy output is byte-identical across both transports for all four prompts, and the
  pipeline is byte-identical to single-node — the network split is lossless (success
  criterion 1)
- **The network round-trip is the entire TB→WiFi delta.** Everything else is within
  noise between transports; round-trip goes 23.4 ms → 52.5 ms and throughput halves.
  That ~30 ms WiFi round-trip is dominated by the 256 KB logits coming *back*, not the
  6 KB activation going out — the logits-asymmetry cost flagged in the step-2 plan,
  now measured as the load-bearing term over WiFi
- **The pipeline bubble, measured firsthand:** local (~19–21 ms) and worker (~22 ms)
  run strictly serially, so each machine sits idle roughly half of every token. Even on
  Thunderbolt (near-zero network) two-node throughput (~21 tok/s) is below the single-
  process single-node loop (~33 tok/s) — the two devices don't overlap. This is the
  "before" picture step 3/4 speculative decoding is meant to recover by keeping both
  busy
- Raw per-token CSVs in `bench_out/{tb,wifi}_{10,50,100,250}.csv`

# Two-Node Pipeline — 24B Model (Fits on Neither Machine Alone)

Model: Mistral-Small-24B-Instruct-2501-4bit (13.3 GB, 40 layers, **untied** lm_head)  
Coordinator: 2023 MacBook Air, M2, 8GB RAM (embed + layers 0–10 + sampling)  
Worker: borrowed Mac, 16GB RAM (layers 10–40 + final norm + lm_head), `0.0.0.0:9000`  
Transports: Thunderbolt (`10.0.0.2`) and WiFi (`192.168.4.77`), both to `:9000`  
Date: August 17–18, 2026

First model too large for either machine to load alone (13.3 GB exceeds both RAM budgets),
run across both via the pipeline split. Enabled by two changes to the MLX layer:
- `mlx_lm.load(..., lazy=True)` — weights are memory-mapped and only materialize when a
  layer is actually run, so each node's resident RAM scales with its layer slice, not the
  full model. Measured directly: `mx.load` adds 0 GB until a tensor is `eval`'d; a node
  running only its slice materializes only that slice. The old `lazy=False` default called
  `mx.eval(model.parameters())` at load, materializing all 40 layers (~13 GB) and OOMing
  the 8 GB node outright.
- `decode_logits` now respects `tie_word_embeddings`. Untied Mistral has a separate
  `lm_head`; the previous code hardcoded the tied `embed_tokens.as_linear` path and
  produced garbage logits. Only a real run caught this — the 3B is tied, so every offline
  check passed.

## Finding the split: the 24B sits right at the pair's RAM edge

Total weights (13.3 GB) barely fit across 8 GB + 16 GB, so the *balance* of the split
decides whether either machine swaps. The 8 GB coordinator thrashes above ~3.5 GB; the
16 GB worker above ~10 GB. Tuning, each measured on hardware:

| Split (coord/worker) | Coord RAM | Worker RAM | Result |
|---|---:|---:|---|
| 12 / 28 | 4.14 GB | ~9.2 GB | **coordinator swaps** → ~0.03 tok/s (~38 s/tok) |
| 8 / 32 | 2.92 GB | 10.45 GB | coord OK, but **worker swaps** under load (per-tok max 9.6 s) → 1.9 tok/s |
| **10 / 30** | **3.53 GB** | ~9.8 GB (est.) | **both stable** → steady 6.2 tok/s (TB) |

10/30 is the sweet spot: small enough on the 8 GB Air, below the worker's threshold on the
16 GB Mac. `lazy=True` is what makes any of this possible (RAM ∝ layers run); the split
just balances the two edges. Full 13.3 GB still downloads to each machine's disk — disk is
not the constraint, RAM is.

## Throughput sweep (10/30 split, 100 generated tokens, greedy)

`steady` excludes a first-few-token prefill transient (see notes); it is the real
per-token rate. `raw` includes it and is dominated by it only at the longest prompt.

| Prompt tokens | TB steady | TB raw | WiFi steady | WiFi raw | ttft TB / WiFi |
|--------------:|----------:|-------:|------------:|---------:|---------------:|
|            12 |       6.3 |    6.3 |         5.0 |      5.0 |    0.6 / 0.9 s |
|            59 |       6.2 |    5.7 |         4.8 |      4.6 |    1.3 / 1.7 s |
|           110 |       6.3 |    5.7 |         4.9 |      4.3 |    3.3 / 3.2 s |
|           256 |       6.2 |    1.2 |         4.9 |      1.2 |    5.3 / 6.3 s |

Steady-state throughput is **flat across prompt length** (KV cache working) —
**~6.2 tok/s Thunderbolt, ~4.9 tok/s WiFi**, TB ≈ 1.3× WiFi.

## Per-token decode latency (median, steady state, ms)

| local (coord, 10L) | worker (30L + logits) | network | round-trip | sample |
|------:|-------:|--------:|-----------:|-------:|
| 42 | 109 | TB 1 / WiFi 26 | TB 110 / WiFi 137 | 3 |

## Wire sizes

| Transfer | Size |
|---|---|
| Decode activation (1 token, out) | ~10 KB (hidden 5120 × fp16) |
| Logits (1 position, back) | 262,144 B (~256 KB; vocab 131,072 × fp16) |
| Prefill activation, 12-token prompt (out) | 122,880 B (~120 KB) |

## Notes
- **Correct:** coherent greedy output across two machines — *"A KV cache is a data
  structure that stores key-value pairs in memory for quick access and retrieval."* A
  model that loads on neither Mac alone now runs on both.
- **RAM ∝ layers run, proven on hardware** (10/30: coord 3.53 GB, worker ~9.8 GB vs 13.3 GB
  full). This is the result that makes the whole project premise hold — splitting buys
  model size.
- **Thunderbolt ≈ 1.3× WiFi**, and the entire delta is the network stage: 1 ms (TB) vs
  26 ms (WiFi) per token, which is the 256 KB logits coming *back*. Everything else
  (local 42 ms, worker 109 ms, sample 3 ms) is transport-independent. Logits dominate the
  wire vs the ~10 KB activation out, as at 3B.
- **The P250 "collapse" (raw 1.2 tok/s) is a startup transient, not steady state.** After
  the large 256-token prefill, the first ~3 decode tokens stall 12–30 s each (`local` max
  30 s) as the **8 GB coordinator** pages under the prefill's memory spike, then recover to
  ~42 ms. Over a 100-token run those 3 tokens are ~69 of 83 s → the *average* tanks, but
  steady state is unaffected (6.2 tok/s). The **worker** stays stable throughout (mean
  125 ms). So each machine has a distinct memory edge: the worker's is fixed by the split;
  the coordinator's shows only after a big prefill and amortizes over longer generations.
- **Lazy-loading tradeoff:** saves RAM but front-loads materialization into ttft / the
  first tokens. Steady-state is unaffected once weights are resident.
- **The pipeline bubble persists:** local (42 ms) and worker (109 ms) run strictly
  serially, so each machine idles part of every token. Even on Thunderbolt it's ~6 tok/s —
  the "before" picture speculative decoding (step 3/4) is meant to recover.
- Raw per-token CSVs in `bench_out/mistral24b/{tb,wifi}_{10,50,100,250}.csv`;
  sweep driver `run_sweep_24b.sh`.

---

# Step 3 — Speculative Decoding, Milestone A (single node, greedy)

Draft: Llama-3.2-1B-Instruct-4bit · Target: Llama-3.2-3B-Instruct-4bit (tied)
Hardware: 2023 MacBook Air, M2, 8 GB RAM
Date: August 20, 2026

Both models in one machine, two MLX server processes (target :8765, draft :8766).
Draft proposes K tokens (one `/draft` call: primes `cur`, greedy-generates x_1..x_K,
sync-feeds x_K); target verifies all K in **one** batched forward and returns per-position
greedy argmax; accept/reject + symmetric KV-cache trim run in the Rust loop
(`--spec-k`, `--draft-model`). Leviathan greedy degenerate case: accept x_i iff
`argmax(p_i) == x_i`.

## Correctness — byte-identical gate ✅

Temp-0 spec output is **byte-identical** to the plain single-node greedy transcript across
**3 prompts** (KV-cache explainer + count, a Python function, a prime list) × **K ∈
{1,2,4}**. This exercises all three rollback regimes — accept-none (`trim K`), partial
(`trim K−a`), all-accepted (`trim 0`, relies on the draft's sync-fed x_K keeping the two
caches symmetric). The offset-aware causal mask on the K-wide verify pass
(`create_attention_mask(x, cache[0])`, mirroring mlx-lm's own `LlamaModel.__call__`) is
what makes each verify position reproduce sequential decode exactly. This is the milestone
deliverable.

## Performance — K sweep (prompt: KV-cache explainer + count, 128 tok cap)

| K | accept α | mean accepted / verify | tok/s | rel. | draft ms/round | verify ms/round |
|---|---|---|---|---|---|---|
| baseline (greedy) | — | 1.00 | **35.4** | 1.00× | — | — |
| 1 | 0.806 | 1.77 | 32.1 | 0.91× | 24.5 | 28.5 |
| 2 | 0.739 | 2.39 | 33.4 | 0.94× | 29.4 | 39.5 |
| 4 | 0.641 | 3.44 | 29.5 | 0.83× | 50.0 | 62.4 |
| 6 | 0.456 | 3.67 | 21.8 | 0.62× | 71.1 | 92.4 |

α and accepted-tokens/verify behave exactly as theory predicts (α falls as K grows;
accepted/verify rises but sub-linearly). α is higher on structured/repetitive prompts
(prime list α=0.907 at K=2) than prose.

## Notes

- **Single-node spec is a *net loss* here (0.6–0.94×), and the diagnosis is clean.** Verify
  batches efficiently — the K-token target forward scales at only ~13 ms/extra token, so at
  K=4 verify costs ~18 ms per *accepted* token; **verify-only would be ~1.5×.** But the 1B
  **draft's compute (24–71 ms/round) more than eats that gain.** Per round at K=4:
  draft 50 + verify 62 = 112 ms for 3.44 tokens = 32.6 ms/tok vs baseline 28.2 ms/tok.
- **Why this is the *expected* single-node result, not a bug.** The "verify K for the price
  of 1" premise holds when a forward is latency-bound by weight loading (big GPUs); on the
  M2 Air a 3B-4bit forward is already fast, so K tokens add real marginal cost and the draft
  is pure overhead. This is precisely why the interesting regime is **distributed**
  (Milestone B): there the ~30 ms/token logits round-trip dominates, and spec amortizes it
  over ~αK emitted tokens per round-trip — the draft compute is hidden behind network cost.
- **It also motivates the zero-compute draft** (doc Tier 2 #4, prompt-lookup / n-gram): with
  draft cost → 0, the K=4 verify-only path projects to ~1.5× *on a single node* and needs no
  second model on the 8 GB Air. Strong next A/B.
- Architecture reuses the existing pipeline: verify = `/embed`+`/forward`+`/argmax`, the same
  chain the decode loop already walks, so it generalizes to the two-node split with no
  rewrite. New primitives: `PartialModel.{draft_generate, greedy_all, trim}`, endpoints
  `/draft` `/argmax` `/trim`, Rust `run_spec_loop`.

---

# Step 3 — Speculative Decoding, Milestone A (temp>0, **exact**)

Same hardware / model pair, Aug 20 2026. Extends greedy spec to full Leviathan sampling.

The draft now **samples** K tokens from `q` at temperature T (seeded) and keeps its
distributions; the target returns per-position `p` at T (`/verify_probs`, fp32); the
**accept/reject + residual resampling runs on the draft server** (`/accept`), which is where
`q` lives — this is the B-aligned placement (in Milestone B the draft is the
coordinator-local model and §1.3 has the accept test run there). Rust plumbs the tensors and
coordinates the same symmetric `K−a` trim as greedy. Accept x_j iff `r < min(1, p_j(x_j)/q_j(x_j))`;
at first reject resample from `norm(max(0, p_j − q_j))`; if all K accepted, free bonus from `p_K`.
New flag `--spec-seed` (per-round seed = base + round); token #0 is a seeded target sample.

## Exactness — proof + statistical gate ✅

**Algebraic:** for the first emitted position, `P(emit y) = min(q(y),p(y)) + max(0, p(y)−q(y)) = p(y)` —
the output is exactly the target distribution, independent of the draft. So the correctness gate
is statistical (not byte-level, as with greedy).

**Empirical:** fix a context, compute the target's exact `p*`, run N spec rounds, compare the
empirical distribution of the emitted token to `p*` via total-variation distance — against a
noise floor = expected TV of N *direct* multinomial draws from `p*`. Spec is exact iff
`TV(spec) ≈ TV(direct)`.

| T | N | TV(spec, p*) | noise floor | ratio | verdict |
|---|---|---|---|---|---|
| 0.3 | 400 | 0.0052 | 0.0245 | 0.21 | within floor ✅ |
| 0.8 | 400 | 0.0216 | 0.0421 | 0.51 | within floor ✅ |
| 1.0 | 700 | 0.0267 | 0.0507 | 0.53 | within floor ✅ |

Every ratio ≤ 1 — spec's deviation from `p*` is no larger than pure sampling noise. (At T=1.0,
N=200 gave a misleading ratio 1.68; raising N→700 dropped it to 0.53, confirming it was small-N
noise on the flatter high-temp distribution, exactly as the proof predicts.)

## Reproducibility + α

- **Seeded → reproducible:** same `--spec-seed` twice → byte-identical transcript; different seed
  → different text. (Greedy determinism was byte-identical to baseline; temp>0 determinism is
  per-seed.)
- **α across 5 seeds** (T=0.8, K=4, real story-continuation prompt): 0.44, 0.49, 0.54, 0.63, 0.49
  — stable ~0.5, spread is just different sampled trajectories. mean accepted/verify ≈ 2.7–3.4.
- Perf tracks the greedy finding: ~24 tok/s at T=0.8 K=4 (draft compute still dominates
  single-node; the win is distributed / zero-compute-draft, per the greedy notes).

## Notes / honesty

- **A ships the full `p` [K+1, vocab] fp32** target→coordinator each round (localhost, cheap —
  correctness milestone). Milestone B replaces this with §1.3's lazy return (K scalars `p_i(x_i)`
  + one full distribution at the reject point) so the slow WiFi link isn't hit with K×256 KB.
- New primitives: `PartialModel.{draft_sample, spec_accept, verify_probs}`, seeded `sample_token`;
  endpoints `/draft_sample` `/verify_probs` `/accept`; fp32 tensor support; Rust
  `{draft_sample, verify_probs, accept, sample_seeded}` + temp>0 branch in `run_spec_loop`.

## Next

§1.3 lazy-logits return, then distributed **Milestone B** (`Trim` TCP frame,
draft-on-coordinator, verify across the pipeline). The single-node greedy + exact-sampling
machinery here is the reusable base for both.

---

# Step 3 — Greedy-draft speculative sampling (lazy accept-on-verifier)

Same hardware / model pair, Aug 23 2026. New `--draft-temp` flag (default **0**);
`--temperature` is the target temp as before. This is §1.3's lazy return, made trivial.

**Idea.** Force the **draft to temp 0** (argmax). Then `q_j` is a point mass on the drafted
token `m_j`, and Leviathan collapses: accept iff `r < p_j(m_j)` (a scalar), and the reject
resample is just **`p_j` with `m_j` zeroed and renormalized** (`norm(max(0,p−q))` with a
point-mass `q`). Marginal is still exactly `p` (`P(emit m_j)=p_j(m_j)`,
`P(emit y≠m_j)=p_j(y)`). Because the verifier already knows the drafted ids, it runs the whole
accept+resample itself and **returns just `(accepted, final_token)` — no `q`, no distribution
on the wire.** For `--draft-temp > 0` the old sampled-draft path (ships full `p`) still runs.

## Deliverable 1 — network return per token: greedy vs shipping a distribution

**Why temp 0 is special:** the reject resample `norm(max(0,p−q))` needs `q`. For any
`draft-temp>0`, `q` lives on the draft, so ≥1 full distribution must cross the wire (the
verifier can't resample alone). At `draft-temp=0`, `q` is known, so the verifier resamples and
returns one token id. Measured `verify_ret` per round: **greedy = 8 B**, sampled =
**2,565,120 B** (= (K+1)·V·4, K=4, V=128 256, fp32) — confirmed on the wire.

Per **emitted** token (÷ measured mean-accepted/verify `T_round`), for our V=128 256 pair:

| target T | sampled T_round | naive (K+1 fp32) | lazy min (1 fp32) | lazy min (1 fp16) | greedy |
|---|---|---|---|---|---|
| 0.3 | 2.97 | 843 KB/tok | 169 KB/tok | 84 KB/tok | ~3 B/tok |
| 0.7 | 2.88 | 870 KB/tok | 174 KB/tok | 87 KB/tok | ~4 B/tok |
| 1.0 | 2.71 | 924 KB/tok | 185 KB/tok | 92 KB/tok | ~4 B/tok |

Greedy eliminates essentially the entire return: ~**850 KB/token** vs the current impl, ~**170
KB/token** vs the best *exact* lazy scheme (fp32, 1 distribution). For scale, the pipeline's
forward activation is ~10 KB/token — the returned distribution is 8–90× larger.

**Generalizes by vocab** (one distribution = `V·b`; per-token = `V·b / T_round`):

| Model family | V | fp16 dist | fp32 dist |
|---|---|---|---|
| Llama-2 / Mistral-7B | 32,000 | 62 KB | 125 KB |
| GPT-2 / Pythia | ~50,300 | 98 KB | 196 KB |
| Llama-3.x (ours) | 128,256 | 250 KB | 501 KB |
| Mistral-Small-24B (repo) | 131,072 | 256 KB | 512 KB |
| Qwen2.5 | 151,936 | 297 KB | 594 KB |
| Gemma-2/3 | 256,000 | 500 KB | 1000 KB |

**WiFi projection** (from the two-node data: 256 KB fp16 ≈ 28 ms one-way), pure return time:

| scheme | return/round | return/token |
|---|---|---|
| naive (5 fp32 dists, current) | ~274 ms | **~96 ms/tok** |
| lazy fp32 (1 dist) | ~55 ms | ~19 ms/tok |
| lazy fp16 (1 dist) | ~27 ms | ~10 ms/tok |
| **greedy (2 ints)** | ~0 ms | **~0 ms/tok** |

Since per-token compute here is ~50 ms, the distribution return is a 0.2–2× latency tax on a
WiFi link that greedy erases entirely.

## Deliverable 2 — acceptance rate vs draft temperature

Prompt fixed, K=4, `--spec-seed 1`, warm. α (and mean accepted/verify) as `--draft-temp` sweeps
under each target `T`:

| target T \ draft T | 0 (greedy) | 0.3 | 0.7 | 1.0 |
|---|---|---|---|---|
| 0.3 | α 0.372 / 2.44 | **0.523 / 2.97** | 0.382 / 2.50 | 0.436 / 2.71 |
| 0.7 | α 0.290 / 2.16 | 0.356 / 2.38 | **0.485 / 2.88** | 0.470 / 2.88 |
| 1.0 | α 0.245 / 1.98 | 0.129 / 1.48 | 0.375 / 2.50 | **0.429 / 2.71** |

α is **maximized when the draft temp matches the target temp** (bold), consistent with
`α = 1 − TV(p_target@T, q_draft@Td)`. Greedy draft (Td=0) always proposes the mode, so it
gives up α vs the temp-matched draft — the price of zero `q`-bandwidth. A mismatched non-zero
draft temp can be *worse* than greedy (e.g. T=1.0, Td=0.3 → α 0.129). **(Single prompt / seed —
generalized below.)**

### Multi-prompt validation (5 prompts × 3 seeds × 3 temps, greedy vs temp-matched)

The greedy-vs-matched α **gap is not a constant** — it grows with temperature and shrinks on
structured text:

| target T | mean gap | std | min | max |
|---|---|---|---|---|
| 0.3 | **−0.018** | 0.100 | −0.200 | 0.151 |
| 0.7 | +0.134 | 0.151 | −0.093 | 0.345 |
| 1.0 | +0.257 | 0.171 | −0.032 | 0.626 |
| all | +0.124 | 0.183 | −0.200 | 0.626 |

At low temp greedy essentially **ties** matched (the draft's argmax is usually the target's
mode); at high temp it gives up ~0.26. Per prompt (avg over seeds/temps): repetitive **+0.059**,
reasoning **+0.068**, factual +0.146, prose +0.162, code **+0.185** — structured/repetitive text
costs greedy almost nothing; creative/code costs most. So the earlier single-prompt "~0.18" was
a mid-temp/prose point, not representative.

## The honest single-node result

On **localhost** the 2.5 MB return costs only ~5 ms/round, so the α loss dominates and greedy
is a **net loss**: tok/s at (T, greedy vs temp-matched) = 0.3: 20.6 vs 23.9 · 0.7: 18.3 vs 23.6
· 1.0: 16.6 vs 22.0. **Greedy draft is a bandwidth optimization, not a single-node one** — its
whole payoff is the network return it removes, which only bites on a real link. Projected on
WiFi it flips: greedy ≈ 19 tok/s vs ~7 (naive) / ~17–20 (lazy), while being simpler (no `q`
state, accept fully on the verifier, exact — no fp16-on-wire approximation).

## Break-even vs the lazy-logits method (how much α greedy may give up)

Greedy trades acceptance rate for return bandwidth. When is that trade worth it vs the exact
lazy method (draft-temp>0, returns K scalars + 1 distribution)? Per-token time in a synchronous
round (emits `T=αK+1` tokens): `t=(C+R)/(αK+1)`, with `C` the per-round cost common to both
(draft + target forward + activation send) and `R` the worker→coordinator return — the only
difference (greedy `R_g≈0`; lazy `R_L≈` one distribution). Setting `t_greedy ≤ t_lazy`:

```
Δα_max = [ ΔR / (C + R_L) ] · (α_L + 1/K),   ΔR = R_L − R_g ≈ one-distribution transfer
```

Greedy's tolerable α **drop** scales with the fraction of the round spent shipping the
distribution. With measured inputs (`C≈117 ms/round`, `α_L≈0.5`, `K=4`; transfer from the
two-node data: 256 KB fp16 ≈ 28 ms WiFi, ~1 ms Thunderbolt):

| link / dtype | ΔR | Δα_max | vs observed drop ≈ 0.18 |
|---|---|---|---|
| localhost / Thunderbolt | 0.5–2 ms | 0.003–0.013 | greedy loses badly |
| WiFi, fp16 lazy (approx) | 28 ms | 0.145 | ~tie / slight loss |
| **WiFi, fp32 lazy (exact)** | 56 ms | **0.243** | **greedy wins** |

**Threshold:** against the *exact* lazy method (which must ship fp32), greedy tolerates an α
drop of up to **~0.24** on WiFi (≈ one accepted token/round, `K·Δα≈1`) — collapsing to ~0.01 on
Thunderbolt/localhost. Sensitivity to compute: `Δα_max` = 0.36 at `C=60 ms`, 0.16 at
`C=200 ms` (WiFi fp32) — faster compute favors greedy.

**Verdict, evaluated per-case on the multi-prompt sweep** (projecting WiFi per-token time from
each case's *measured* `T_round`, `C=117 ms`, fp32 return 56 ms / fp16 28 ms):

| vs lazy | greedy wins | mean proj tok/s (greedy vs lazy) |
|---|---|---|
| exact (fp32) | **34/45 (76%)** | **25.6 vs 20.0** |
| approx (fp16) | 26/45 (58%) | 25.6 vs 23.8 |

Per temp (WiFi proj tok/s, greedy / lazy-fp32 / lazy-fp16): T=0.3 **31.4 / 20.8 / 24.8** · T=0.7
**25.5 / 20.1 / 24.0** · T=1.0 **19.8 / 19.0 / 22.7**. So **greedy clearly wins vs the exact
lazy method at low–mid temperature and on structured prompts**, narrowing to a tie at T=1.0
(where its α gap balloons to ~0.26). Against an *approximate* fp16 lazy it's roughly a wash
(and loses at T=1.0) — but greedy stays exact and simpler (no `q` state, accept fully on the
verifier). On Thunderbolt/localhost greedy loses everywhere (return is nearly free). Caveat:
assumes a synchronous pipeline; async overlap (Milestone C) would hide `R` and shrink greedy's
edge.

## Correctness

- **Exactness of the greedy-draft path — confirmed.** Empirical first-emitted-token
  distribution vs the target's exact `p*`, TV distance as a z-score against the
  direct-multinomial noise floor: T=0.3 z=0.60 (N=800), T=1.0 z=1.70 (N=800), T=0.8 z=0.39
  (N=2000). T=0.8 read z=2.22 at N=800 but **fell to 0.39 at N=2000** — a real bias grows with
  √N, so shrinking confirms it was sampling noise. (`verify_accept` is algebraically identical
  to the validated `spec_accept` with a point-mass `q`: `min(1,p/1)=p`, `max(0,p−δ)` = `p`
  with the drafted id zeroed.)
- **Reproducible:** same `--spec-seed` → byte-identical transcript.
- **No regression:** `--temperature 0` still byte-identical to `baseline.out`.

New primitives: `PartialModel.verify_accept`; endpoint `/verify_accept`; Rust
`verify_accept` + `--draft-temp` + `verify_ret_bytes` instrumentation.

## Next

Milestone B makes this the worker-side primitive: worker runs verify+accept and returns
`(a, final)` over TCP — the greedy-draft lazy return *is* the distributed win projected above.

---

# Step 3 — Milestone B: distributed speculative decoding (correctness, loopback)

Draft runs entirely on the coordinator; the target is split coordinator-shard (layers 0..14) +
worker-shard (14..28). Each round: draft K, forward all K+1 through the local shard, ship once to
the worker, accept, roll back **all three** KV caches (draft, local shard, worker-over-TCP) by
`k−a`. New TCP frames: `Trim`, `Verify`/`VerifyResult`, `LogitsAt` (+ a small `aux: Vec<u32>`
side-channel on every frame). Dispatch: `--mode coordinator --spec-k K --draft-model <url>`.

**All gates run as 4 processes on one M2 Air over loopback TCP** — a functional/correctness check,
not a perf headline (all shards contend for one GPU). The two-Mac Thunderbolt/WiFi benchmark is
deferred until after these pass.

## Correctness

- **Greedy (temp 0) — byte-identical to the non-spec two-node pipeline** (`run_coordinator`) across
  3 prompts × K∈{1,2,4} (all 9 exact, 80 tokens each). Byte-identity across varying K (hence varying
  accept counts) is also the cache-rollback proof: any `trim` desync across the three caches would
  corrupt later tokens. Worker returns its per-position **argmax ids** — the greedy lazy return is
  `(K+1)×4 = 20 B/round`, never the 256 KB logits the plain pipeline ships per token.
- **Sampled (temp>0) — byte-identical to single-node `run_spec_loop`** across 2 prompts ×
  {K∈{2,4}, T∈{0.8,0.3}}, seeded (`--spec-seed`). The two-phase lazy return
  (`accept_scalars`→`logits_at`→`resample_at`) replicates `spec_accept`'s exact RNG stream, so
  equality is byte-level, not just statistical.
- **Reproducible:** same `--spec-seed` → identical transcript.

## Wire (measured on loopback via the round stats)

- Greedy K=4: α≈0.60, mean_accepted/verify≈3.3, **round_trips/tok≈0.30** (one worker round-trip per
  ~3.3 emitted tokens vs exactly 1.0 for the non-spec pipeline), verify_ret **20 B/round**.
- Sampled K=4 T=0.8: α≈0.50, mean_accepted/verify≈2.95, return **≈513 KB/round** = exactly **one**
  full fp32 distribution (128256×4 B) + the K+1 scalars — vs the naive K+1 distributions (~2.5 MB).
  The lazy return collapses the sampled return to one distribution per *round* instead of per *token*.

New primitives: TCP `Trim`/`Verify`/`VerifyResult`/`LogitsAt` frames + `Frame.aux`;
`run_spec_coordinator`, `verify_exchange`, `logits_at_exchange`, worker Verify/LogitsAt/Trim arms;
`PartialModel.verify_scalars`/`logits_at`/`accept_scalars`/`resample_at`; endpoints
`/verify_scalars`, `/logits_at`, `/accept_scalars`, `/resample_at`; client methods to match.

## Next

Two-Mac benchmark over Thunderbolt + WiFi: tok/s, α, round-trips/token, return-bytes/token vs the
two-node pipeline baseline — the "worker idle time drops from ~65% to ~20%" number. Then the K-sweep
per transport (optimal K rises as bandwidth falls), and fp16-on-the-wire for the sampled distribution.