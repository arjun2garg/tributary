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