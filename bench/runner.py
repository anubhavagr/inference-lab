"""Workload runner — drives an engine over a prompt set, captures stats, emits JSON."""                                                           
from __future__ import annotations                                                                                                                
import json                                                                                                                                       
import sys                                                                                                                                        
from dataclasses import asdict, dataclass                                                                                                         
from pathlib import Path                                                                                                                          
                                                                                                                                                
from rich.console import Console                                                                                                                  
from rich.table import Table                                                                                                                      
                                                                                                                                                
from engines.base import Engine                                                                                                                   
                                                                                                                                                
console = Console()                                                                                                                               
              
# Mix of short and longer prompts so prompt-processing cost varies.                                                                               
PROMPTS = [     
  "Explain gradient descent in three sentences.",                                                                                               
  "What is the difference between TCP and UDP?",                                                                                                
  "Write a Python function to check if a string is a palindrome.",                                                                              
  "Summarize the plot of Hamlet in one paragraph.",                                                                                             
  "Why is the sky blue? Answer in two sentences.",                                                                                              
  "What does the softmax function do, and why is it called softmax?",                                                                           
  "Name three causes of the 2008 financial crisis.",                                                                                            
  "Describe how HTTPS establishes a secure connection.",                                                                                        
  "Explain attention in transformers. Use one paragraph.",                                                                                      
  "What is the CAP theorem? Give a concrete example.",                                                                                          
  # longer prompts — exercise prefill                                                                                                           
  "Describe the architecture of a transformer model, including attention, "                                                                     
  "layer normalization, and the feed-forward sublayer. Use four paragraphs "                                                                    
  "and be specific about what each component does and why it matters.",                                                                         
  "Compare and contrast supervised learning, unsupervised learning, and "                                                                       
  "reinforcement learning. For each, describe the typical training signal, "                                                                    
  "the kind of problems it solves, and one practical failure mode.",                                                                            
]                                                                                                                                                 
                                                                                                                                                
                                                                                                                                                
@dataclass                                                                                                                                        
class RunConfig:                                                                                                                                  
  engine_name: str                                                                                                                              
  model_id: str                                                                                                                                 
  n_warmup: int = 3                                                                                                                             
  n_measured: int = 20                                                                                                                          
  max_tokens: int = 256                                                                                                                         
  temperature: float = 0.0                                                                                                                      
                                                                                                                                                
                                                                                                                                                
def percentile(values: list[float], p: float) -> float:                                                                                           
  """Linear-interp percentile — no numpy dependency."""                                                                                         
  if not values:                                                                                                                                
      return float("nan")
  s = sorted(values)                                                                                                                            
  k = (len(s) - 1) * p                                                                                                                          
  f = int(k)                                                                                                                                    
  c = min(f + 1, len(s) - 1)                                                                                                                    
  return s[f] + (s[c] - s[f]) * (k - f)                                                                                                         
                                                                                                                                                
                                                                                                                                                
def run_bench(engine: Engine, prompts: list[str], cfg: RunConfig) -> dict:                                                                        
  console.print(f"[bold cyan]Loading engine:[/bold cyan] {engine.name} ({cfg.model_id})")                                                       
  engine.load()                                                                                                                                 
                                                                                                                                                
  try:                                                                                                                                          
      console.print(f"[dim]warmup ({cfg.n_warmup} requests, not measured)...[/dim]")                                                            
      for i in range(cfg.n_warmup):                                                                                                             
          engine.generate(prompts[i % len(prompts)], max_tokens=cfg.max_tokens, temperature=cfg.temperature)                                    
                                                                                                                                                
      results = []                                                                                                                              
      console.print(f"[dim]measured ({cfg.n_measured} requests)...[/dim]")                                                                      
      for i in range(cfg.n_measured):                                                                                                           
          r = engine.generate(prompts[i % len(prompts)], max_tokens=cfg.max_tokens, temperature=cfg.temperature)                                
          results.append(r)                                                                                                                     
          console.print(                                                                                                                        
              f"  [{i+1:>2}/{cfg.n_measured}] "                                                                                                 
              f"TTFT {r.stats.ttft_ms:6.0f}ms  "                                                                                                
              f"TPOT {r.stats.tpot_ms:5.1f}ms  "                                                                                                
              f"{r.stats.output_tokens:>3} tok"                                                                                                 
          )                                                                                                                                     
                                                                                                                                                
      ttfts = [r.stats.ttft_ms for r in results]                                                                                                
      tpots = [r.stats.tpot_ms for r in results]
      totals = [r.stats.total_ms for r in results]
      out_toks = [r.stats.output_tokens for r in results]
      prompt_toks = [r.stats.prompt_tokens for r in results]
                                                                                                                                                
      return {
          "config": asdict(cfg),                                                                                                                
          "n_measured": len(results),
          "prompt_tokens_mean": sum(prompt_toks) / len(prompt_toks),
          "output_tokens_mean": sum(out_toks) / len(out_toks),
          "ttft_ms": {
              "p50": percentile(ttfts, 0.50),
              "p99": percentile(ttfts, 0.99),
              "mean": sum(ttfts) / len(ttfts),
          },
          "tpot_ms": {
              "p50": percentile(tpots, 0.50),                                                                                                   
              "mean": sum(tpots) / len(tpots),                                                                                                  
          },                                                                                                                                    
          "total_ms": {                                                                                                                         
              "p50": percentile(totals, 0.50),                                                                                                  
              "p99": percentile(totals, 0.99),                                                                                                  
          },                                                                                                                                    
          "throughput_tok_per_s": sum(out_toks) / (sum(totals) / 1000),                                                                         
          "peak_rss_gb": max(r.stats.peak_rss_gb for r in results),                                                                             
          "raw": [asdict(r.stats) for r in results],                                                                                            
      }                                                                                                                                         
  finally:                                                                                                                                      
      engine.unload()                                                                                                                           
                                                                                                                                                
                                                                                                                                                
def save_summary(summary: dict, out_path: Path) -> None:                                                                                          
  out_path.parent.mkdir(parents=True, exist_ok=True)                                                                                            
  out_path.write_text(json.dumps(summary, indent=2))                                                                                            
  console.print(f"[green]saved:[/green] {out_path}")                                                                                            
                                                                                                                                                
                                                                                                                                                
def print_table(rows: list[dict]) -> None:                                                                                                        
  t = Table(title="Inference Lab — results")                                                                                                    
  t.add_column("engine")                                                                                                                        
  t.add_column("model", style="cyan")                                                                                                           
  t.add_column("TTFT p50", justify="right")                                                                                                     
  t.add_column("TTFT p99", justify="right")                                                                                                     
  t.add_column("TPOT p50", justify="right")                                                                                                     
  t.add_column("tok/s", justify="right")                                                                                                        
  t.add_column("RSS GB", justify="right")                                                                                                       
  for r in rows:                                                                                                                                
      t.add_row(                                                                                                                                
          r["config"]["engine_name"],                                                                                                           
          r["config"]["model_id"],                                                                                                              
          f"{r['ttft_ms']['p50']:.0f}",                                                                                                         
          f"{r['ttft_ms']['p99']:.0f}",                                                                                                         
          f"{r['tpot_ms']['p50']:.1f}",                                                                                                         
          f"{r['throughput_tok_per_s']:.1f}",                                                                                                   
          f"{r['peak_rss_gb']:.2f}",                                                                                                            
      )                                                                                                                                         
  console.print(t)                        
