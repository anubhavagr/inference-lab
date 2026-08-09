"""Async concurrency-sweep runner."""
from __future__ import annotations
import asyncio
import time
from dataclasses import asdict, dataclass
from statistics import median

from rich.console import Console

from bench.runner import percentile
from engines.base import Engine, GenerateResult

console = Console()


@dataclass
class SweepConfig:
  engine_name: str
  model_id: str
  concurrency_levels: tuple[int, ...] = (1, 4, 8, 16, 32)
  n_per_level: int = 16         # requests to fire at each level
  n_warmup: int = 4
  max_tokens: int = 256
  temperature: float = 0.0


async def _call_async(engine: Engine, prompt: str, cfg: SweepConfig, loop, executor) -> GenerateResult:
  # Both engines' generate() are blocking — push to a worker thread.
  return await loop.run_in_executor(
      executor,
      lambda: engine.generate(prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature),
  )


def _sample_prompt(prompts: list[str], i: int) -> str:
  return prompts[i % len(prompts)]


def run_sweep(engine: Engine, prompts: list[str], cfg: SweepConfig) -> dict:
  import concurrent.futures
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)

  console.print(f"[bold cyan]Loading engine:[/bold cyan] {engine.name} ({cfg.model_id})")
  engine.load()

  try:
      # warmup
      console.print(f"[dim]warmup ({cfg.n_warmup})...[/dim]")
      loop.run_until_complete(asyncio.gather(*[
          _call_async(engine, _sample_prompt(prompts, i), cfg, loop, concurrent.futures.ThreadPoolExecutor(max_workers=4))
          for i in range(cfg.n_warmup)
      ]))

      levels: list[dict] = []
      for c in cfg.concurrency_levels:
          console.print(f"[bold]concurrency = {c}[/bold]  ({cfg.n_per_level} requests)")
          executor = concurrent.futures.ThreadPoolExecutor(max_workers=c)

          t_wall_start = time.perf_counter()
          results = loop.run_until_complete(asyncio.gather(*[
              _call_async(engine, _sample_prompt(prompts, i), cfg, loop, executor)
              for i in range(cfg.n_per_level)
          ]))
          wall_s = time.perf_counter() - t_wall_start
          executor.shutdown(wait=True)

          ttfts = [r.stats.ttft_ms for r in results]
          tpots = [r.stats.tpot_ms for r in results]
          out_toks = [r.stats.output_tokens for r in results]

          aggregate_tok_per_s = sum(out_toks) / wall_s

          levels.append({
              "concurrency": c,
              "wall_seconds": wall_s,
              "n_requests": len(results),
              "output_tokens_total": sum(out_toks),
              "ttft_ms_p50": percentile(ttfts, 0.50),
              "ttft_ms_p99": percentile(ttfts, 0.99),
              "tpot_ms_p50": percentile(tpots, 0.50),
              # Aggregate: total output tokens / wall-clock. This is the
              # metric that matters for serving — "how many tokens per
              # second can ONE device push when N users hit it?"
              "aggregate_tok_per_s": aggregate_tok_per_s,
              "per_request_tok_per_s": median([r.stats.output_tokens / (r.stats.total_ms / 1000) for r in results]),
          })
          console.print(
              f"  → wall {wall_s:.1f}s, "
              f"TTFT p50 {levels[-1]['ttft_ms_p50']:.0f}ms, "
              f"aggregate {aggregate_tok_per_s:.1f} tok/s"
          )

      return {
          "config": asdict(cfg),
          "levels": levels,
          "raw_count": len(levels) * cfg.n_per_level,
      }
  finally:
      engine.unload()
      loop.close()

