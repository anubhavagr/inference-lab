"""Engine registry — importable from worker processes without pulling both backends."""
from __future__ import annotations

# key -> display name (safe to import anywhere; pulls no heavy deps)
ENGINES = {
    "mlx":   "mlx-lm",
    "llama": "llama.cpp",
}


def get_engine_class(key: str):
    """Lazily import only the engine backend asked for."""
    if key == "mlx":
        from engines.mlx_engine import MlxEngine
        return MlxEngine
    if key == "llama":
        from engines.llama_cpp_engine import LlamaCppEngine
        return LlamaCppEngine
    raise KeyError(f"unknown engine key: {key!r}")
