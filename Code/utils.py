import tiktoken

model2max_context = {
    'Qwen3-8B': 16384,
    'Qwen3-4B': 16384,
    'Qwen3-1.7B':16384,
    'Meta-Llama-3.1-8B-Instruct': 16384,
    'Meta-Llama-3.1-70B-Instruct-AWQ-INT4': 16384,
    'GLM-4-9B-0414': 16384,
    'Qwen2.5-7B-Instruct': 16384,
    'qwen2_5-14b-instruct': 16384,
    'qwen2_5-32b-instruct': 16384,
    'Qwen3-0.6B': 16384,
    
}

class OutOfQuotaException(Exception):
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


def num_tokens_from_string(string: str, model_name: str) -> int:
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        import tiktoken.load
        original_get = requests.get
        requests.get = session.get
        
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            result = len(encoding.encode(string))
            return result
        finally:
            requests.get = original_get
            
    except Exception as e:
        print(f"Warning: tiktoken failed, using fallback token counting. Error: {e}")
        if not string:
            return 0
            
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', string))
        english_chars = len(re.findall(r'[a-zA-Z]', string))
        other_chars = len(string) - chinese_chars - english_chars
        
        estimated_tokens = chinese_chars * 2 + english_chars * 0.25 + other_chars * 0.3
        return max(int(estimated_tokens), 1)