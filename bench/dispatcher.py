"""K-instance dispatcher: K process-isolated engines draining one request queue.

Phase 1 showed a single engine instance serializes every request through its
lock — aggregate throughput is flat from concurrency 1 to 32. This benchmark
builds the mitigation (a pool of independent instances) and measures where it
lands. Decode streams the full weights every token, so K instances contend for
the same unified-memory bandwidth: sublinear scaling would mean the memory
wall, not the lock, is the true ceiling.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import resource
import sys
import time
from dataclasses import asdict, dataclass
from statistics import median

from rich.console import Console

from bench.runner import percentile

console = Console()


@dataclass
class DispatcherConfig:
    engine_name: str
    model_id: str
    instance_counts: tuple[int, ...] = (1, 2, 3)
    requests_per_level: int = 16
    warmup_per_instance: int = 2
    max_tokens: int = 256
    temperature: float = 0.0


def _peak_rss_gb() -> float:
    """Peak RSS of this process. macOS reports ru_maxrss in bytes, Linux in KB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def _serve(engine_key: str, model_path: str, instance_index: int,
           request_queue, result_queue, ready_queue, config: DispatcherConfig) -> None:
    """Worker process: own one engine instance, drain the request queue."""
    from engines import get_engine_class
    from bench.runner import PROMPTS

    engine = get_engine_class(engine_key)(model_path)
    engine.load()
    for step in range(config.warmup_per_instance):
        engine.generate(PROMPTS[(instance_index + step) % len(PROMPTS)],
                        max_tokens=config.max_tokens, temperature=config.temperature)
    ready_queue.put(instance_index)

    # time.perf_counter is mach_absolute_time on macOS — comparable across processes.
    while (request := request_queue.get()) is not None:
        request_id, prompt, submitted_at = request
        queue_wait_ms = (time.perf_counter() - submitted_at) * 1000
        result = engine.generate(prompt, max_tokens=config.max_tokens,
                                 temperature=config.temperature)
        result_queue.put(("result", request_id, result.stats.__dict__, queue_wait_ms))

    engine.unload()
    result_queue.put(("exit", instance_index, _peak_rss_gb()))
    result_queue.close()
    result_queue.join_thread()  # flush the feeder, or the exit message is lost


def _shutdown(workers, request_queue) -> None:
    for _ in workers:
        request_queue.put(None)
    for worker in workers:
        worker.join(timeout=60)
        if worker.is_alive():
            console.print(f"[red]{worker.name} did not exit — terminating[/red]")
            worker.terminate()
            worker.join(timeout=5)


def run_dispatch(engine_key: str, model_path, prompts: list[str],
                 config: DispatcherConfig) -> dict:
    spawn = mp.get_context("spawn")  # fork + Metal is unsafe on macOS
    levels = []

    for instance_count in config.instance_counts:
        console.print(f"[bold cyan]K = {instance_count}[/bold cyan] "
                      f"({config.requests_per_level} requests, {config.engine_name})")
        request_queue, result_queue, ready_queue = spawn.Queue(), spawn.Queue(), spawn.Queue()
        workers = [
            spawn.Process(target=_serve,
                          args=(engine_key, str(model_path), index,
                                request_queue, result_queue, ready_queue, config))
            for index in range(instance_count)
        ]
        for worker in workers:
            worker.start()

        try:
            for index in range(instance_count):
                ready_queue.get(timeout=600)  # blocks through model load + warmup

            started_at = time.perf_counter()
            for request_id in range(config.requests_per_level):
                request_queue.put((request_id, prompts[request_id % len(prompts)],
                                   time.perf_counter()))

            results, waits = {}, {}
            while len(results) < config.requests_per_level:
                kind, request_id, *payload = result_queue.get(timeout=600)
                if kind == "result":
                    results[request_id] = payload[0]
                    waits[request_id] = payload[1]
            wall_seconds = time.perf_counter() - started_at
        finally:
            _shutdown(workers, request_queue)

        worker_rss_gb = 0.0
        try:
            while True:
                kind, _, peak = result_queue.get_nowait()
                if kind == "exit":
                    worker_rss_gb += peak
        except queue.Empty:
            pass

        ordered = [results[request_id] for request_id in range(config.requests_per_level)]
        ttfts = [stats["ttft_ms"] for stats in ordered]
        tpots = [stats["tpot_ms"] for stats in ordered]
        output_tokens = [stats["output_tokens"] for stats in ordered]
        ttfts_incl_queue = [stats["ttft_ms"] + waits[request_id]
                            for request_id, stats in enumerate(ordered)]

        levels.append({
            "k": instance_count,
            "wall_seconds": wall_seconds,
            "n_requests": config.requests_per_level,
            "output_tokens_total": sum(output_tokens),
            "ttft_ms_p50": percentile(ttfts, 0.50),
            "ttft_ms_p99": percentile(ttfts, 0.99),
            "ttft_incl_queue_ms_p50": percentile(ttfts_incl_queue, 0.50),
            "queue_wait_ms_p50": percentile(list(waits.values()), 0.50),
            "tpot_ms_p50": percentile(tpots, 0.50),
            "aggregate_tok_per_s": sum(output_tokens) / wall_seconds,
            "per_request_tok_per_s": median(
                [stats["output_tokens"] / (stats["total_ms"] / 1000) for stats in ordered]),
            "worker_rss_gb_sum": worker_rss_gb,
        })
        level = levels[-1]
        console.print(f"  → wall {wall_seconds:.1f}s, "
                      f"agg {level['aggregate_tok_per_s']:.1f} tok/s, "
                      f"TTFT+queue p50 {level['ttft_incl_queue_ms_p50']:.0f}ms, "
                      f"RSS {worker_rss_gb:.1f} GB")

    baseline = next((level for level in levels if level["k"] == 1), levels[0])
    for level in levels:
        level["scaling_vs_k1"] = level["aggregate_tok_per_s"] / baseline["aggregate_tok_per_s"]

    return {"config": asdict(config), "levels": levels,
            "raw_count": len(levels) * config.requests_per_level}
