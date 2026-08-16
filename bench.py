"""Run the inference-lab bench.

      python bench.py <engine> <model_path>                        # single-stream
      python bench.py sweep <engine> <model_path>                  # concurrency sweep
      python bench.py dispatch <engine> <model_path> [--instances 1,2,3]  # K-instance dispatcher
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from bench.runner import RunConfig, PROMPTS, run_bench, save_summary, print_table
from bench.async_runner import SweepConfig, run_sweep
from bench.dispatcher import DispatcherConfig, run_dispatch
from engines import ENGINES, get_engine_class


def run_single(engine_key: str, model_path: str) -> None:
  model_path = Path(model_path)
  if not model_path.exists():
      print(f"error: {model_path} not found"); sys.exit(1)
  cfg = RunConfig(
      engine_name=ENGINES[engine_key],
      model_id=model_path.name,
      n_warmup=3,
      n_measured=20,
      max_tokens=256,
  )
  engine = get_engine_class(engine_key)(model_path)
  summary = run_bench(engine, PROMPTS, cfg)
  save_summary(summary, Path(f"results/raw/{cfg.model_id}_{cfg.engine_name}.json"))
  print_table([summary])


def _print_sweep(summary: dict) -> None:
  c = Console()
  t = Table(title=f"Sweep — {summary['config']['engine_name']} on {summary['config']['model_id']}")
  t.add_column("conc")
  t.add_column("wall (s)", justify="right")
  t.add_column("TTFT p50", justify="right")
  t.add_column("TTFT p99", justify="right")
  t.add_column("agg tok/s", justify="right")
  t.add_column("per-req tok/s", justify="right")
  for lv in summary["levels"]:
      t.add_row(
          str(lv["concurrency"]),
          f"{lv['wall_seconds']:.1f}",
          f"{lv['ttft_ms_p50']:.0f}",
          f"{lv['ttft_ms_p99']:.0f}",
          f"{lv['aggregate_tok_per_s']:.1f}",
          f"{lv['per_request_tok_per_s']:.1f}",
      )
  c.print(t)


def run_sweep_cmd(engine_key: str, model_path: str) -> None:
  model_path = Path(model_path)
  if not model_path.exists():
      print(f"error: {model_path} not found"); sys.exit(1)
  cfg = SweepConfig(
      engine_name=ENGINES[engine_key],
      model_id=model_path.name,
      concurrency_levels=(1, 4, 8, 16, 32),
      n_per_level=16,
      n_warmup=4,
      max_tokens=256,
  )
  engine = get_engine_class(engine_key)(model_path)
  summary = run_sweep(engine, PROMPTS, cfg)
  out = Path(f"results/raw/sweep_{cfg.model_id}_{cfg.engine_name}.json")
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(summary, indent=2))
  _print_sweep(summary)


def _print_dispatch(summary: dict) -> None:
  c = Console()
  t = Table(title=f"Dispatcher — {summary['config']['engine_name']} on {summary['config']['model_id']}")
  t.add_column("K")
  t.add_column("wall (s)", justify="right")
  t.add_column("TTFT p50", justify="right")
  t.add_column("TTFT+queue p50", justify="right")
  t.add_column("agg tok/s", justify="right")
  t.add_column("×K=1", justify="right")
  t.add_column("RSS sum GB", justify="right")
  for lv in summary["levels"]:
      t.add_row(
          str(lv["k"]),
          f"{lv['wall_seconds']:.1f}",
          f"{lv['ttft_ms_p50']:.0f}",
          f"{lv['ttft_incl_queue_ms_p50']:.0f}",
          f"{lv['aggregate_tok_per_s']:.1f}",
          f"{lv['scaling_vs_k1']:.2f}",
          f"{lv['worker_rss_gb_sum']:.1f}",
      )
  c.print(t)


def run_dispatch_cmd(engine_key: str, model_path: str, instance_counts: tuple[int, ...]) -> None:
  model_path = Path(model_path)
  if not model_path.exists():
      print(f"error: {model_path} not found"); sys.exit(1)
  cfg = DispatcherConfig(
      engine_name=ENGINES[engine_key],
      model_id=model_path.name,
      instance_counts=tuple(sorted(instance_counts)),
  )
  summary = run_dispatch(engine_key, model_path, PROMPTS, cfg)
  out = Path(f"results/raw/dispatch_{cfg.model_id}_{cfg.engine_name}.json")
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(summary, indent=2))
  _print_dispatch(summary)


def _parse_instance_counts(raw: str) -> tuple[int, ...]:
  try:
      return tuple(int(value) for value in raw.split(",") if value.strip())
  except ValueError:
      print(f"error: --instances expects comma-separated counts, got {raw!r}")
      sys.exit(1)


def main() -> None:
  if len(sys.argv) < 3:
      print(__doc__)
      sys.exit(1)

  if sys.argv[1] == "sweep":
      if len(sys.argv) < 4 or sys.argv[2] not in ENGINES:
          print(f"usage: python bench.py sweep <{'|'.join(ENGINES)}> <model_path>")
          sys.exit(1)
      run_sweep_cmd(sys.argv[2], sys.argv[3])
  elif sys.argv[1] == "dispatch":
      arguments = sys.argv[2:]
      instance_counts = (1, 2, 3)
      if "--instances" in arguments:
          index = arguments.index("--instances")
          if index + 1 >= len(arguments):
              print("error: --instances needs a value, e.g. --instances 1,2,3")
              sys.exit(1)
          instance_counts = _parse_instance_counts(arguments[index + 1])
          del arguments[index:index + 2]
      if len(arguments) < 2 or arguments[0] not in ENGINES:
          print(f"usage: python bench.py dispatch <{'|'.join(ENGINES)}> <model_path> [--instances 1,2,3]")
          sys.exit(1)
      run_dispatch_cmd(arguments[0], arguments[1], instance_counts)
  elif sys.argv[1] in ENGINES:
      run_single(sys.argv[1], sys.argv[2])
  else:
      print(f"unknown engine '{sys.argv[1]}'. choices: {', '.join(ENGINES)}")
      sys.exit(1)


if __name__ == "__main__":
  main()
