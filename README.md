# inference-lab

A reproducible LLM-serving benchmark for Apple Silicon and cloud GPU.

> *Can a MacBook Pro usefully serve a 7B LLM — and at what concurrency does it stop being useful?*

Phase 1 answers this on an M4 Pro for Qwen2.5-7B-Instruct at 4-bit across `mlx-lm` and `llama.cpp`, single-stream and concurrency sweep. Phase 2 extends the same workload to a cloud H100 with vLLM and TensorRT-LLM.

**Read the write-up:** [`docs/writeup-1.md`](docs/writeup-1.md) — MLX vs llama.cpp on Apple Silicon: a fair LLM-serving benchmark.

**Or browse it on the portfolio:** [anubhavagr.github.io/posts/inference-lab](https://anubhavagr.github.io/posts/inference-lab.html)

## Headline numbers

Single-stream on M4 Pro, Qwen2.5-7B-Instruct @ 4-bit:

| Engine    | TTFT p50 | TPOT p50 | tok/s | RSS GB |
|-----------|---------:|---------:|------:|-------:|
| mlx-lm    |   228 ms | 26.1 ms  | 37.1  |  4.42  |
| llama.cpp |   100 ms | 21.7 ms  | 45.3  |  4.76  |

Concurrency sweep (1 → 32): both engines flatline within ±4% of single-stream throughput. Single-instance serving on Apple Silicon is a ceiling, not a scaling curve.

## Reproduce

```bash
git clone https://github.com/anubhavagr/inference-lab
cd inference-lab
uv venv && source .venv/bin/activate
uv pip install mlx-lm llama-cpp-python matplotlib rich

hf download mlx-community/Qwen2.5-7B-Instruct-4bit --local-dir models/qwen25-7b-mlx-4bit
hf download bartowski/Qwen2.5-7B-Instruct-GGUF --include "*Q4_K_M*" --local-dir models/qwen25-7b-gguf-q4k

python bench.py mlx   models/qwen25-7b-mlx-4bit
python bench.py llama models/qwen25-7b-gguf-q4k/Qwen2.5-7B-Instruct-Q4_K_M.gguf
python bench.py sweep mlx   models/qwen25-7b-mlx-4bit
python bench.py sweep llama models/qwen25-7b-gguf-q4k/Qwen2.5-7B-Instruct-Q4_K_M.gguf
python -m bench.plots
```

## Repository layout

```
bench/
  runner.py            # single-stream driver — warmup, percentiles, JSON
  async_runner.py      # concurrency sweep
  plots.py             # matplotlib charts from saved JSON
  PROMPTS              # 12-prompt workload (in runner.py)
engines/
  base.py              # Engine interface
  mlx_engine.py        # mlx-lm wrapper, per-instance lock
  llama_cpp_engine.py  # llama-cpp-python wrapper, per-instance lock
results/
  raw/*.json           # gitignored — regenerate locally
  README.md            # publishable summary table
docs/
  methodology.md       # the fairness contract
  writeup-1.md         # the blog post
  img/                 # the three charts
```

## Status

- [x] **Phase 1** — Apple Silicon bench (this repo, current state)
- [ ] **Phase 2** — cloud burst (H100 + vLLM + TensorRT-LLM)
- [ ] **Phase 3** — `serve/` (OpenAI-compatible API, semantic cache, cost teardown)

## License

MIT — see [`LICENSE`](LICENSE) when added. Numbers and prose in `docs/writeup-1.md` are
attributed via the repo URL if cited elsewhere.
