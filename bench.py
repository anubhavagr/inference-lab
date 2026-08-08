"""Run the inference-lab bench: python bench.py <model_path>"""
from __future__ import annotations                                                                                                                
import sys                                                                                                                                        
from pathlib import Path                                                                                                                          
                                                                                                                                                
from bench.runner import RunConfig, PROMPTS, run_bench, save_summary, print_table                                                                 
from engines.mlx_engine import MlxEngine
from engines.llama_cpp_engine import LlamaCppEngine                                                                                               
                                                                                                                                                
ENGINES = {                                                                                                                                       
  "mlx":   MlxEngine,                                                                                                                           
  "llama": LlamaCppEngine,                                                                                                                      
}                                                                                                                                                 
                                                                                                                                                
                                                                                                                                                
def main():     
  if len(sys.argv) < 3 or sys.argv[1] not in ENGINES:
      print(f"usage: python bench.py <{'|'.join(ENGINES)}> <model_path>")                                                                       
      sys.exit(1)                                                                                                                               
                                                                                                                                                
  engine_key = sys.argv[1]                                                                                                                      
  model_path = Path(sys.argv[2])                                                                                                                
  if not model_path.exists():                                                                                                                   
      print(f"error: {model_path} not found")                                                                                                   
      sys.exit(1)                                                                                                                               
                                                                                                                                                
  cfg = RunConfig(                                                                                                                              
      engine_name=ENGINES[engine_key].name,                                                                                                     
      model_id=model_path.name,                                                                                                                 
      n_warmup=3,                                                                                                                               
      n_measured=20,                                                                                                                            
      max_tokens=256,                                                                                                                           
  )                                                                                                                                             
                                                                                                                                                
  engine = ENGINES[engine_key](model_path)                                                                                                      
  summary = run_bench(engine, PROMPTS, cfg)                                                                                                     
  save_summary(summary, Path(f"results/raw/{cfg.model_id}_{cfg.engine_name}.json"))                                                             
  print_table([summary])                                                                                                                        
                                                                                                                                                
                                                                                                                                                
if __name__ == "__main__":
  main()

