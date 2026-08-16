# Results       
                                                                                                                                                
All numbers measured on Apple M4 Pro, macOS, single stream, greedy decode,                                                                        
`max_tokens=256`, 3 warmup + 20 measured requests per data point. Prompts
are a fixed 12-prompt set of varying length (5 short, 5 medium, 2 long).                                                                          
See [../docs/methodology.md](../docs/methodology.md) for full fairness contract.                                                                  
                                                                                                                                                
## Qwen2.5-7B-Instruct @ 4-bit                                                                                                                    
                                                                                                                                                
| Engine    | TTFT p50 | TTFT p99 | TPOT p50 | tok/s | RSS GB |                                                                                   
|-----------|---------:|---------:|---------:|------:|-------:|
| mlx-lm    |   228 ms |   332 ms | 26.1 ms  |  37.1 |  4.42  |                                                                                   
| llama.cpp |   100 ms |   174 ms | 21.7 ms  |  45.3 |  4.76  |                                                                                   
                                                                                                                                                
**Headline:** On M4 Pro at 4-bit, llama.cpp outperforms mlx-lm on every                                                                           
serving-relevant metric — TTFT is 2.3× faster, decode throughput 22%                                                                              
higher, at an 8% memory cost.                                                                                                                     
              
Raw JSON per run: `raw/<model>_<engine>.json`.

## Dispatcher — K process-isolated instances (`bench.py dispatch`)

16 requests offered at once per level; K instances, each in its own
process, drain one shared queue. Same workload contract as the sweep.

**llama.cpp**

| K | agg tok/s | ×K=1 | TPOT p50 | per-req tok/s | TTFT p50 | RSS sum GB |
|---|----------:|-----:|---------:|--------------:|---------:|-----------:|
| 1 | 45.3 | 1.00 | 21.7 ms | 45.4 |  99 ms |  4.7 |
| 2 | 47.8 | 1.06 | 41.1 ms | 23.9 | 132 ms |  9.5 |
| 3 | 49.3 | 1.09 | 60.1 ms | 16.5 | 160 ms | 14.2 |

**mlx-lm**

| K | agg tok/s | ×K=1 | TPOT p50 | per-req tok/s | TTFT p50 | RSS sum GB |
|---|----------:|-----:|---------:|--------------:|---------:|-----------:|
| 1 | 55.6 | 1.00 | 17.4 ms | 55.8 | 163 ms |  4.3 |
| 2 | 60.2 | 1.08 | 32.4 ms | 30.1 | 236 ms |  8.6 |
| 3 | 60.7 | 1.09 | 48.1 ms | 20.4 | 312 ms | ~13* |

\* mlx worker RSS under-reports under memory pressure (9.1–11.2 GB
observed across runs; llama.cpp scales linearly — see methodology).

**Headline:** instances do not scale on Apple Silicon. Aggregate
throughput gains ≤9% for 3× the memory, while each stream's decode rate
degrades near-linearly (TPOT roughly ×K). A single 4-bit 7B instance
already streams ~250 GB/s of the M4 Pro's ~273 GB/s unified-memory
bandwidth — the bus saturates before the second instance helps. The only
lever that beats this wall is continuous batching, which reads the
weights once per forward pass for the whole batch.

Raw JSON per run: `raw/dispatch_<model>_<engine>.json`.

