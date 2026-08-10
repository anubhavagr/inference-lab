"""Generate the three result charts from saved JSON files."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

# Match the portfolio palette roughly.
mpl.rcParams.update({
  "figure.facecolor":    "#f1ece0",
  "axes.facecolor":      "#fbf7ef",
  "axes.edgecolor":      "#e0d6c4",
  "axes.labelcolor":     "#211e17",
  "xtick.color":         "#544e44",
  "ytick.color":         "#544e44",
  "axes.titleweight":    "bold",
  "axes.titlecolor":     "#211e17",
  "font.family":         ["Inter", "DejaVu Sans", "sans-serif"],
  "font.size":           11,
  "axes.grid":           True,
  "grid.color":          "#ebe3d2",
  "grid.linewidth":      0.8,
})

MLX_COLOR   = "#b0613f"   # clay accent
LLAMA_COLOR = "#5b6f8c"   # cool counterpoint


def load_results(pattern: str, root: Path = Path("results/raw")) -> list[dict]:
  return [json.loads(p.read_text()) for p in sorted(root.glob(pattern))]


def plot_single_stream(out_path: Path) -> None:
  files = {
      "mlx-lm":    "qwen25-7b-mlx-4bit_mlx-lm.json",
      "llama.cpp": "Qwen2.5-7B-Instruct-Q4_K_M.gguf_llama.cpp.json",
  }
  rows = {}
  for label, fname in files.items():
      p = Path("results/raw") / fname
      if not p.exists():
          print(f"skip: {p} not found")
          return
      d = json.loads(p.read_text())
      rows[label] = d

  metrics = [
      ("TTFT p50 (ms)",  lambda d: d["ttft_ms"]["p50"]),
      ("TPOT p50 (ms)",  lambda d: d["tpot_ms"]["p50"]),
      ("Throughput (tok/s)", lambda d: d["throughput_tok_per_s"]),
      ("Peak RSS (GB)",  lambda d: d["peak_rss_gb"]),
  ]

  fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
  labels = list(rows.keys())
  colors = [MLX_COLOR, LLAMA_COLOR]

  for ax, (title, fn) in zip(axes, metrics):
      vals = [fn(d) for d in rows.values()]
      bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="#211e17", linewidth=0.6)
      ax.set_title(title)
      ax.margins(y=0.18)
      for b, v in zip(bars, vals):
          ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                  ha="center", va="bottom", fontsize=10, color="#211e17")
      # rotate x labels
      ax.set_xticklabels(labels, rotation=0, fontsize=9)

  fig.suptitle("Single-stream — Qwen2.5-7B-Instruct @ 4-bit on M4 Pro",
               fontsize=13, y=1.02)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
  print(f"wrote {out_path}")


def plot_sweep(metric_key: str, y_label: str, title: str, out_path: Path) -> None:
  files = {
      "mlx-lm":    f"sweep_qwen25-7b-mlx-4bit_mlx-lm.json",
      "llama.cpp": f"sweep_Qwen2.5-7B-Instruct-Q4_K_M.gguf_llama.cpp.json",
  }
  series = {}
  for label, fname in files.items():
      p = Path("results/raw") / fname
      if not p.exists():
          print(f"skip: {p} not found")
          return
      d = json.loads(p.read_text())
      series[label] = [(lv["concurrency"], lv[metric_key]) for lv in d["levels"]]

  fig, ax = plt.subplots(figsize=(7.5, 4.2))
  for (label, pts), color in zip(series.items(), [MLX_COLOR, LLAMA_COLOR]):
      xs = [p[0] for p in pts]
      ys = [p[1] for p in pts]
      ax.plot(xs, ys, marker="o", color=color, label=label, linewidth=2, markersize=7)
      for x, y in zip(xs, ys):
          ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                      xytext=(0, 9), ha="center", fontsize=9, color=color)

  ax.set_xlabel("concurrent requests")
  ax.set_ylabel(y_label)
  ax.set_title(title)
  ax.set_xticks(sorted({x for pts in series.values() for x, _ in pts}))
  ax.legend(frameon=False, loc="best")
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
  print(f"wrote {out_path}")


def main():
  out_dir = Path("docs/img")
  out_dir.mkdir(parents=True, exist_ok=True)
  plot_single_stream(out_dir / "single_stream.png")
  plot_sweep("aggregate_tok_per_s", "aggregate tok/s",
             "Throughput vs concurrency — Qwen2.5-7B-4bit on M4 Pro",
             out_dir / "sweep_throughput.png")
  plot_sweep("ttft_ms_p50", "TTFT p50 (ms)",
             "TTFT vs concurrency — Qwen2.5-7B-4bit on M4 Pro",
             out_dir / "sweep_ttft.png")


if __name__ == "__main__":
  main()

