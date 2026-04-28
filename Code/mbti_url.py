import openai
import backoff
import requests
import time
import json
from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError
from .utils import OutOfQuotaException, AccessTerminatedException
from .utils import num_tokens_from_string, model2max_context

support_models = ['Qwen3-8B','Qwen3-4B','Qwen3-1.7B','Meta-Llama-3.1-8B-Instruct','Meta-Llama-3.1-70B-Instruct-AWQ-INT4']


MODEL_CONFIGS = {
    'local': {
        'base_url': 'http://localhost:8000', 
        'endpoint': '/v1/completions',  
        'headers': {'Content-Type': 'application/json'},
        'api_type': 'completion'  
    }
}

MODEL_PROVIDER_MAP = {
    'Qwen3-8B': 'local', 
    'Qwen3-4B': 'local',
    'Qwen3-1.7B': 'local',
    'Meta-Llama-3.1-8B-Instruct': 'local',
    'Meta-Llama-3.1-70B-Instruct-AWQ-INT4': 'local',
}

def get_model_config(model_name):
    
    if model_name in MODEL_PROVIDER_MAP:
        provider = MODEL_PROVIDER_MAP[model_name]
        print(f"Match: {model_name} -> {provider}")
        return MODEL_CONFIGS[provider]
    
    for pattern, provider in MODEL_PROVIDER_MAP.items():
        if pattern.endswith('-') and model_name.startswith(pattern):
            print(f"match: {model_name} -> {provider} (pattern: {pattern})")
            return MODEL_CONFIGS[provider]
    
    return MODEL_CONFIGS['local']


class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, api_key: str, sleep_time: float = 0) -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.api_key = api_key
        self.sleep_time = sleep_time
        self.meta_prompt = None
        self.enable_thinking = False

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError), max_tries=20)
    def query(self, messages: "list[dict]", max_tokens: int, api_key: str, temperature: float) -> str:
        print(f"model_name={self.model_name}")
        
        time.sleep(self.sleep_time)

        model_config = get_model_config(self.model_name)
        base_url = model_config['base_url']
        endpoint = model_config['endpoint']
        url = f"{base_url}{endpoint}"
        
        print(f"URL: {url}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        is_local_model = 'localhost' in base_url or '127.0.0.1' in base_url or 'local' in self.model_name.lower()
        print(f"DEBUG: {is_local_model}")

        if is_local_model:
            prompt = self.messages_to_prompt(messages)
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
        else:
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
        

        try:
            print(f"{self.model_name}")
            
            session = requests.Session()
            session.trust_env = False
            
            timeout = 120 if is_local_model else 30
            
            response = session.post(
                url,
                headers=headers,
                json=data,
                timeout=timeout,
                verify=False
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    if is_local_model:
                        if 'choices' in result and len(result['choices']) > 0:
                            gen = result['choices'][0]['text']
                            print(f"✅ Received response from local model: {gen[:100]}...")
                            return gen
                        else:
                            error_msg = f"Unexpected local model response format: {result}"
                            print(f"❌ {error_msg}")
                            return "Local model response format error"
                    else:
                        # Cloud ChatCompletion API response format
                        if 'choices' in result and len(result['choices']) > 0:
                            gen = result['choices'][0]['message']['content']
                            print(f"✅ Received response from cloud model: {gen[:100]}...")
                            return gen
                        else:
                            error_msg = f"Unexpected cloud model response format: {result}"
                            print(f"❌ {error_msg}")
                            return "Cloud model response format error"
                            
                except json.JSONDecodeError as e:
                    print("❌ Failed to parse JSON response!")
                    print(f"🔧 Full response content: {response.text}")
                    print(f"🔧 Error details: {e}")
                    return "JSON parsing failed: response is not valid JSON"
                    
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return f"API error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            return "Request timed out"
            
        except requests.exceptions.ConnectionError:
            print("❌ Connection error")
            return "Connection error"
            
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Request failed: {str(e)}"


    def messages_to_prompt(self, messages):
        prompt = ""
        for message in messages:
            role = message['role']
            content = message['content']
            
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    def set_meta_prompt(self, meta_prompt: str):
        self.meta_prompt = meta_prompt
        self.memory_lst.append({"role": "system", "content": f"{meta_prompt}"})

    def add_event(self, event: str):
        self.memory_lst.append({"role": "user", "content": f"{event}"})

    def add_memory(self, memory: str):
        self.memory_lst.append({"role": "assistant", "content": f"{memory}"})
        print(f"----- {self.name} -----\n{memory}\n")

    def ask(self, temperature: float = None, api_key: str = ''):
        num_context_token = sum([num_tokens_from_string(m["content"], self.model_name) for m in self.memory_lst])
        
        if self.model_name in model2max_context:
            max_context = model2max_context[self.model_name]
        else:
            max_context = 4096
        
        max_token = max_context - num_context_token
        
        max_token = max(max_token, 100)
        
        
        return self.query(self.memory_lst, max_token, api_key=self.api_key, temperature=temperature if temperature else self.temperature)
    
    def ask_single_turn(self, event_prompt: str, meta_prompt: str = None) -> str:
        self.memory_lst = []

        system_prompt = meta_prompt if meta_prompt is not None else self.meta_prompt
        if system_prompt is not None:
            self.memory_lst.append({"role": "system", "content": system_prompt})

        self.memory_lst.append({"role": "user", "content": event_prompt})

        response = self.ask()

        return response
