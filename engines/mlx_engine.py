"""mlx-lm engine wrapper — Apple Silicon native."""                                                                                               
from __future__ import annotations                                                                                                                
import threading
import resource                                                                                                                                   
import time                                                                                                                                       
from pathlib import Path                                                                                                                          
                                                                                                                                                
import mlx.core as mx                                                                                                                             
from mlx_lm import load, stream_generate                                                                                                          
                                                                                                                                                
from engines.base import Engine, GenerateResult, GenerateStats                                                                                    
                                                                                                                                                
                                                                                                                                                
def _peak_rss_gb() -> float:
  """Peak RSS in GB. macOS reports ru_maxrss in bytes; Linux in KB."""                                                                          
  import sys                                                                                                                                    
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss                                                                                      
  unit = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2                                                                                   
  return rss / unit                                                                                                                             
                                                                                                                                                
                                                                                                                                                
class MlxEngine(Engine):                                                                                                                            
  name = "mlx-lm"                                                                                                                                 
                                                                                                                                                  
  def __init__(self, model_path: str | Path):                                                                                                     
      self.model_path = str(model_path)                                                                                                           
      self.model = None                                                                                                                           
      self.tokenizer = None                                                                                                                       
      # mlx's Metal dispatch is not thread-safe across concurrent                                                                                 
      # stream_generate calls; serialize like llama.cpp.                                                                                          
      self._lock = threading.Lock()                                                                                                               
                                                                                                                                                  
  def load(self) -> None:
      import mlx.core as mx                                                                                                                       
      self.model, self.tokenizer = load(self.model_path)
      mx.eval(self.model.parameters)

  def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerateResult:
      with self._lock:
          return self._generate_locked(prompt, max_tokens, temperature)

  def _generate_locked(self, prompt: str, max_tokens: int, temperature: float) -> GenerateResult:                                                 
      # ← move the entire body of the OLD generate() here, unchanged
      if self.model is None:                                                                                                                      
          raise RuntimeError("Engine not loaded — call .load() first")                                                                            
                                                                                                                                                  
      t_start = time.perf_counter()                                                                                                               
      ttft_ms: float | None = None                                                                                                                
      chunks: list[str] = []                                                                                                                      
      output_tokens = 0                                                                                                                           
      prompt_tokens = 0                                                                                                                           
      mlx_peak_memory = None                                                                                                                      
                                                                                                                                                  
      for response in stream_generate(                                                                                                            
          self.model, self.tokenizer, prompt, max_tokens=max_tokens,                                                                              
      ):                                                                                                                                          
          if ttft_ms is None and response.text:
              ttft_ms = (time.perf_counter() - t_start) * 1000                                                                                    
          chunks.append(response.text)                                                                                                            
          output_tokens += 1                                                                                                                      
          if prompt_tokens == 0:                                                                                                                  
              prompt_tokens = getattr(response, "prompt_tokens", 0)                                                                               
          mlx_peak_memory = getattr(response, "peak_memory", None) or mlx_peak_memory                                                             
                                                                                                                                                  
      total_ms = (time.perf_counter() - t_start) * 1000                                                                                           
      if ttft_ms is None:                                                                                                                         
          ttft_ms = total_ms                                                                                                                      
      decode_ms = total_ms - ttft_ms
      tpot_ms = decode_ms / max(output_tokens - 1, 1)                                                                                             
                                                                                                                                                  
      return GenerateResult(                                                                                                                      
          text="".join(chunks),                                                                                                                   
          stats=GenerateStats(                                                                                                                    
              prompt_tokens=prompt_tokens,                                                                                                        
              output_tokens=output_tokens,                                                                                                        
              ttft_ms=ttft_ms,                                                                                                                    
              total_ms=total_ms,                                                                                                                  
              tpot_ms=tpot_ms,                                                                                                                    
              peak_rss_gb=mlx_peak_memory if mlx_peak_memory is not None else _peak_rss_gb(),                                                     
          ),                                                                                                                                      
      )                                                                                                                                           
                                                                                                                                                  
  def unload(self) -> None:                                                                                                                       
      self.model = None                                                                                                                           
      self.tokenizer = None             
