from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import threading


class ThreadSafeTokenizer:
    """Thread-safe wrapper around HuggingFace tokenizers.

    Maintains one tokenizer instance per thread via thread-local storage,
    eliminating the 'Already Borrowed' panic from Rust-backed fast tokenizers
    under concurrent access.

    All other attributes and methods are forwarded transparently to the
    per-thread tokenizer instance via __getattr__, ensuring full API
    compatibility.
    """
    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self._init_kwargs = kwargs
        self._local = threading.local()

    def _get(self):
        try:
            return self._local.tokenizer
        except AttributeError:
            self._local.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, **self._init_kwargs
            )
            return self._local.tokenizer

    def __call__(self, *args, **kwargs):
        return self._get()(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self._get().encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._get().decode(*args, **kwargs)

    def batch_encode_plus(self, *args, **kwargs):
        return self._get().batch_encode_plus(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self._get().batch_decode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._get(), name)


class RoundProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.st_model = SentenceTransformer(model_id, device=device)
        self.hf_tokenizer = ThreadSafeTokenizer(model_id)
        self.hf_model = AutoModel.from_pretrained(model_id)
        self.hf_model.eval()
        
    def process_round(self, round_data):
        """Transform the 'reason' field of each agent in the round into embeddings."""
        embedded_round = []
        for agent in round_data:
            r = {
                key: agent[key] for key in agent if key != 'reason'
            }
            text = agent['reason']
            
            st_embed = self.st_model.encode(text, device=self.device)
            
            # Now the per-token embeddings
            inputs = self.hf_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.hf_model(**inputs)  # hf_model stays on CPU
            
            # last_hidden_state: [1, num_tokens, 384]
            token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
            token_embeddings = token_embeddings.tolist()

            r['st_embedding'] = st_embed
            r['tk_embedding'] = token_embeddings
            
            embedded_round.append(r)
        return embedded_round
    
class WindowEmbeddings:
    """
    Extended text processing class that adds window embeddings while preserving 
    all existing single-embedding and token-embedding outputs.
    
    This version implements the windowing logic from the WindowProcessor:
    - Tokenizes once
    - Divides tokens into n_windows with overlap
    - Feeds window token tensors directly into hf_model on self.device
    - Performs mean-pooling on the last_hidden_state of each window
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', n_windows=15, overlap=0.5, **kwargs):
        self.device = device
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.st_model = SentenceTransformer(model_id, device=device)
        self.hf_tokenizer = ThreadSafeTokenizer(model_id)
        self.hf_model = AutoModel.from_pretrained(model_id)
        self.hf_model.to(self.device)  # Move HuggingFace model to device
        self.hf_model.eval()
        
        self.n_windows = n_windows
        self.overlap = overlap

    def process_round(self, round_data):
        """Transform the 'reason' (or 'text') field of each agent in the round into embeddings.
        
        This method generates:
        - st_embedding / st_embeddings: existing single embeddings (numpy array)
        - tk_embedding / tk_embeddings: existing token-based embeddings (list of lists)
        - window_embeddings: overlapping token-level window embeddings (list of lists/vectors)
        """
        embedded_round = []
        for agent in round_data:
            r = {
                key: agent[key] for key in agent if key != 'reason'
            }
            
            # Retrieve text — support both 'reason' (GAMMAF default) and 'text'
            text = agent.get('reason', agent.get('text', ''))
            
            # 1. Generate standard SentenceTransformer embedding (single vector)
            st_embed = self.st_model.encode(text, device=self.device)
            r["st_embedding"] = st_embed
            
            # 2. Generate per-token embeddings from HuggingFace model (original tk_embedding logic)
            inputs = self.hf_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
            
            # last_hidden_state: [1, num_tokens, 384]
            tk_embeddings = outputs.last_hidden_state[0].cpu().numpy().tolist()
            r['tk_embedding'] = tk_embeddings
            
            # 3. Generate window-based embeddings using the custom windowing logic
            tokens = self.hf_tokenizer.encode(text, add_special_tokens=False)
            n_tokens = len(tokens)
            
            window_vectors = []
            if n_tokens > 0:
                stride = max(1, int(n_tokens / self.n_windows))
                window_size = int(stride / (1 - self.overlap))
                MAX_LEN = 512
                window_size = min(window_size, MAX_LEN)
                
                for i in range(self.n_windows):
                    start = i * stride
                    end = start + window_size
                    if start >= n_tokens:
                        break
                    
                    window_tokens = tokens[start:end]
                    if not window_tokens:
                        continue
                    
                    window_tokens = window_tokens[:MAX_LEN]
                    
                    input_ids = torch.tensor([window_tokens]).to(self.device)
                    attention_mask = torch.ones_like(input_ids)
                    
                    with torch.no_grad():
                        outputs = self.hf_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                    
                    window_embeddings = outputs.last_hidden_state  # (1, seq_len, hidden)
                    mask = attention_mask.unsqueeze(-1)
                    pooled = (window_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
                    pooled = pooled.squeeze(0)
                    pooled_np = pooled.detach().cpu().numpy().tolist()
                    window_vectors.append(pooled_np)
            
            # Set the values on the response dict
            if 'label' in agent:
                r['label'] = agent['label']
                                        
            r['window_embeddings'] = window_vectors
            
            embedded_round.append(r)
            
        return embedded_round

    def process_single(self, text, label=None):
        """Process a single text string."""
        sample = {'reason': text}
        if label is not None:
            sample['label'] = label
        processed = self.process_round([sample])
        return processed[0]

    def fit(self, X=None, y=None):
        """No-op fit for scikit-learn compatibility."""
        return self

    def transform(self, round_data):
        """scikit-learn compatibility transform."""
        return self.process_round(round_data)

    def fit_transform(self, round_data, y=None):
        """scikit-learn compatibility fit_transform."""
        return self.fit(round_data, y).transform(round_data)
            
            
class WindowsAndText:
    """
    For visualization and debugging of WindowBreaker return
    the embeddings of windows together with the text of each window.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', n_windows=15, overlap=0.5, **kwargs):
        self.device = device
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.st_model = SentenceTransformer(model_id, device=device)
        self.hf_tokenizer = ThreadSafeTokenizer(model_id)
        self.hf_model = AutoModel.from_pretrained(model_id)
        self.hf_model.to(self.device)  # Move HuggingFace model to device
        self.hf_model.eval()
        
        self.n_windows = n_windows
        self.overlap = overlap

    def process_round(self, round_data):
        """Transform the 'reason' (or 'text') field of each agent in the round into embeddings.
        This method generates:
        - st_embedding / st_embeddings: existing single embeddings (numpy array)
        - tk_embedding / tk_embeddings: existing token-based embeddings (list of lists)
        - window_embeddings: overlapping token-level window embeddings (list of lists/vectors)
        - raw_windows: the decoded text corresponding to each window (list of str)
        """
        embedded_round = []
        for agent in round_data:
            r = {
                key: agent[key] for key in agent if key != 'reason'
            }
            
            # Retrieve text — support both 'reason' (GAMMAF default) and 'text'
            text = agent.get('reason', agent.get('text', ''))
            
            # 3. Generate window-based embeddings using the custom windowing logic
            tokens = self.hf_tokenizer.encode(text, add_special_tokens=False)
            n_tokens = len(tokens)
            
            window_vectors = []
            raw_windows = []
            if n_tokens > 0:
                stride = max(1, int(n_tokens / self.n_windows))
                window_size = int(stride / (1 - self.overlap))
                MAX_LEN = 512  # Because the tokenizer we use cannot process longer sequences
                window_size = min(window_size, MAX_LEN)
                
                for i in range(self.n_windows):
                    start = i * stride
                    end = start + window_size
                    if start >= n_tokens:
                        break
                    
                    window_tokens = tokens[start:end]
                    if not window_tokens:
                        continue
                    
                    window_tokens = window_tokens[:MAX_LEN]
                    
                    # Decode the exact token window back into text for reference
                    window_text = self.hf_tokenizer.decode(
                        window_tokens, skip_special_tokens=True
                    )
                    raw_windows.append(window_text)
                    
                    input_ids = torch.tensor([window_tokens]).to(self.device)
                    attention_mask = torch.ones_like(input_ids)
                    
                    with torch.no_grad():
                        outputs = self.hf_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                    
                    window_embeddings = outputs.last_hidden_state  # (1, seq_len, hidden)
                    mask = attention_mask.unsqueeze(-1)
                    pooled = (window_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
                    pooled = pooled.squeeze(0)
                    pooled_np = pooled.detach().cpu().numpy().tolist()
                    window_vectors.append(pooled_np)
            
            # Set the values on the response dict
            if 'label' in agent:
                r['label'] = agent['label']
                                        
            r['window_embeddings'] = window_vectors
            r['raw_windows'] = raw_windows
            
            embedded_round.append(r)
        
        return embedded_round

    def process_single(self, text, label=None):
        """Process a single text string."""
        sample = {'reason': text}
        if label is not None:
            sample['label'] = label
        processed = self.process_round([sample])
        return processed[0]

    def fit(self, X=None, y=None):
        """No-op fit for scikit-learn compatibility."""
        return self

    def transform(self, round_data):
        """scikit-learn compatibility transform."""
        return self.process_round(round_data)

    def fit_transform(self, round_data, y=None):
        """scikit-learn compatibility fit_transform."""
        return self.fit(round_data, y).transform(round_data)