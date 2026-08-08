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
