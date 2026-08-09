# Methodology

  The fairness contract for every measurement in this repo. Any change
  to the contract invalidates prior results — bump the version and re-run.

  ## Hardware
  - Apple MacBook Pro, M4 Pro
  - macOS (latest at time of measurement)
  - Battery at >50%, no other GPU/Metal workloads running

  ## Workload contract
  - **Prompt set:** 12 prompts of varying length (5 short <30 tokens,
    5 medium 30-80 tokens, 2 long >100 tokens). Fixed seed; same set
    across all engines. Source: `bench/runner.py:PROMPTS`.
  - **Decode:** greedy (`temperature=0`), `max_tokens=256`, no system prompt.
  - **Measurement window per data point:**
    - Single-stream: 3 warmup + 20 measured requests, prompts cycled.
    - Sweep: 4 warmup + 16 measured requests per concurrency level.

  ## Reported metrics
  - **TTFT (time-to-first-token):** wall-clock from request submission
    to first non-empty token yielded by the engine's stream.
  - **TPOT (time-per-output-token):** decode-phase only,
    `(total_ms - ttft_ms) / (output_tokens - 1)`.
  - **Aggregate throughput (sweep):** `sum(output_tokens) / wall_seconds`
    across all concurrent requests.
  - **Peak RSS:** reported by the engine (mlx) or `getrusage` (llama.cpp).

  ## Engine versions
  Capture at run time, recorded in each result JSON's `config.engine_name`.
  Pin in `pyproject.toml` once we publish.

  ## What we deliberately do NOT do
  - No temperature >0 — destroys reproducibility.
  - No KV cache预热 across requests — each request is independent.
  - No cross-engine output-equivalence check — engines may produce
    different token sequences at temp=0 (sampler-default differences).
    TPOT/throughput comparisons remain valid because they're decode-rate
    measurements, not output-quality measurements.

  ## Known system-level noise
  A periodic TTFT spike (~1.5× baseline) appears at the same point in
  both engines' runs — likely an OS scheduler / Spotlight / Time Machine
  event. We report p50 as the headline, never mean. p99 captures the
  tail but should not be over-interpreted on n=20.
