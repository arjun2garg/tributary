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