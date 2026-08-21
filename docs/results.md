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

## Next (Milestone A → temp>0 exact, then B)

Greedy only so far. **Next step: temp>0 *exact*** — draft returns full `q` distributions,
residual resampling `norm(max(0, p−q))` at the reject point, seeded-RNG statistical
acceptance check (exactness is statistical, not byte-level, there). Then the §1.3 lazy-logits
return and the distributed Milestone B (`Trim` TCP frame, draft-on-coordinator). The greedy
scaffolding here is the reusable base for all of it.