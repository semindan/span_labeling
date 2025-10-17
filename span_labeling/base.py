
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import List, Dict

class SpanLabeler(ABC):
    def __init__(self, model_name="hermes3:8b"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
        )
        
    def call_api(self, prompt: str) -> str:
        """Call Ollama via OpenAI client"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a precise span labeling system. Follow the output format exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return ""
    
    @abstractmethod
    def format_prompt(self, entry: dict) -> str:
        pass
    
    @abstractmethod
    def parse_response(self, entry: dict) -> List[Dict]:
        pass
    
    def predict(self, entry: dict) -> Dict:
        """Main method"""
        entry["prompt"] = self.format_prompt(entry)
        entry["response"] = self.call_api(entry["prompt"])
        
        try:
            spans = self.parse_response(entry)
            entry["output"] = {
                "success": True,
                "spans": spans,
                "raw_response": entry["response"],
            }
        except Exception as e:
            entry["output"] = {
                "success": False,
                "spans": [],
                "raw_response": entry["response"],
                "error": str(e)
            }
        return entry