"""Engine interface — every backend implements this."""                                                                                           
from __future__ import annotations                                                                                                                
from abc import ABC, abstractmethod                                                                                                               
from dataclasses import dataclass                                                                                                                 
                                                                                                                                                
                                                                                                                                                
@dataclass                                                                                                                                        
class GenerateStats:                                                                                                                              
  prompt_tokens: int                                                                                                                            
  output_tokens: int                                                                                                                            
  ttft_ms: float          # wall-clock time-to-first-token                                                                                      
  total_ms: float         # wall-clock request end-to-end                                                                                       
  tpot_ms: float          # decode-only time per output token                                                                                   
  peak_rss_gb: float                                                                                                                            
                                                                                                                                                
                                                                                                                                                
@dataclass                                                                                                                                        
class GenerateResult:
  text: str                                                                                                                                     
  stats: GenerateStats                                                                                                                          
                                                                                                                                                
                                                                                                                                                
class Engine(ABC):                                                                                                                                
  name: str                                                                                                                                     
                                                                                                                                                
  @abstractmethod                                                                                                                               
  def load(self) -> None: ...                                                                                                                   
                                                                                                                                                
  @abstractmethod
  def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> GenerateResult: ...                                       
                                                                                                                                                
  @abstractmethod
  def unload(self) -> None: ...          
