""""llama.cpp engine wrapper — uses llama-cpp-python with Metal offload."""                                                                          
from __future__ import annotations                                                                                                                  
import resource                                                                                                                                     
import sys                                                                                                                                          
import threading                                                                                                                                    
import time                                                                                                                                         
from pathlib import Path                                                                                                                            
                                                                                                                                                  
from llama_cpp import Llama                                                                                                                         
                                                                                                                                                  
from engines.base import Engine, GenerateResult, GenerateStats


def _peak_rss_gb() -> float:
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  unit = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2                                                                                     
  return rss / unit                                                                                                                               
                                                                                                                                                  
                                                                                                                                                  
class LlamaCppEngine(Engine):                                                                                                                       
  name = "llama.cpp"                                                                                                                              
                                                                                                                                                  
  def __init__(self, model_path: str | Path, n_ctx: int = 4096):                                                                                  
      self.model_path = str(model_path)                                                                                                           
      self.n_ctx = n_ctx                                                                                                                          
      self.llm: Llama | None = None                                                                                                               
      # The llama.cpp Llama class is NOT thread-safe across calls to the                                                                          
      # same instance. Serializing here means concurrency > 1 effectively                                                                         
      # queues on this lock — which is the honest behavior of                                                                                     
      # single-instance serving and the thing we want to measure.                                                                                 
      self._lock = threading.Lock()                                                                                                               
                                                                                                                                                  
  def load(self) -> None:                                                                                                                         
      self.llm = Llama(                                                                                                                           
          model_path=self.model_path,                                                                                                             
          n_ctx=self.n_ctx,                                                                                                                       
          n_gpu_layers=-1,                                                                                                                        
          verbose=False,                                                                                                                          
          seed=0,                                                                                                                                 
      )                                                                                                                                           
                                                                                                                                                  
  def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerateResult:                                             
      if self.llm is None:
          raise RuntimeError("Engine not loaded — call .load() first")                                                                            
                                                                                                                                                  
      with self._lock:                                                                                                                            
          return self._generate_locked(prompt, max_tokens, temperature)                                                                           
                                                                                                                                                  
  def _generate_locked(self, prompt: str, max_tokens: int, temperature: float) -> GenerateResult:
      prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))                                                  
                                                                                                                                                  
      t_start = time.perf_counter()
      ttft_ms: float | None = None                                                                                                                
      chunks: list[str] = []
      completion_tokens = 0

      stream = self.llm.create_chat_completion(
          messages=[{"role": "user", "content": prompt}],
          max_tokens=max_tokens,
          temperature=temperature,
          stream=True,                                                                                                                            
      )                                                                                                                                           
                                                                                                                                                  
      for chunk in stream:                                                                                                                        
          if not chunk.get("choices"):
              continue                                                                                                                            
          if ttft_ms is None:                                                                                                                     
              ttft_ms = (time.perf_counter() - t_start) * 1000                                                                                    
          delta = chunk["choices"][0].get("delta", {})                                                                                            
          content = delta.get("content")                                                                                                          
          if content:                                                                                                                             
              chunks.append(content)                                                                                                              
              completion_tokens += 1                                                                                                              
                                                                                                                                                  
      total_ms = (time.perf_counter() - t_start) * 1000                                                                                           
      if ttft_ms is None:                                                                                                                         
          ttft_ms = total_ms                                                                                                                      
                                                                                                                                                  
      decode_ms = total_ms - ttft_ms                                                                                                              
      tpot_ms = decode_ms / max(completion_tokens - 1, 1)                                                                                         
                                                                                                                                                  
      return GenerateResult(                                                                                                                      
          text="".join(chunks),                                                                                                                   
          stats=GenerateStats(                                                                                                                    
              prompt_tokens=prompt_tokens,                                                                                                        
              output_tokens=completion_tokens,                                                                                                    
              ttft_ms=ttft_ms,                                                                                                                    
              total_ms=total_ms,                                                                                                                  
              tpot_ms=tpot_ms,                                                                                                                    
              peak_rss_gb=_peak_rss_gb(),                                                                                                         
          ),                                                                                                                                      
      )                                                                                                                                           
                                                                                                                                                  
  def unload(self) -> None:                                                                                                                       
      self.llm = None                                                                               
