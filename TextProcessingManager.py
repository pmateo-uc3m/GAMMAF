from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

class RoundProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.st_model = SentenceTransformer(model_id, device=device)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
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
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
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
            # Copy all fields except 'reason' (which will be replaced by 'text' and 'reason')
            # and any existing embedding fields to avoid duplication or conflicts.
            excluded_keys = {'reason', 'st_embedding', 'st_embeddings', 'tk_embedding', 'tk_embeddings', 'window_embeddings'}
            r = {
                key: agent[key] for key in agent if key not in excluded_keys
            }
            
            # Retrieve text — support both 'reason' (GAMMAF default) and 'text'
            text = agent.get('reason', agent.get('text', ''))
            
            # 1. Generate standard SentenceTransformer embedding (single vector)
            st_embed = self.st_model.encode(text, device=self.device)
            
            # 2. Generate per-token embeddings from HuggingFace model (original tk_embedding logic)
            inputs = self.hf_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
            
            # last_hidden_state: [1, num_tokens, 384]
            token_embeddings = outputs.last_hidden_state[0].cpu().numpy().tolist()
            
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
                    
                    token_embeddings = outputs.last_hidden_state  # (1, seq_len, hidden)
                    mask = attention_mask.unsqueeze(-1)
                    pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
                    pooled = pooled.squeeze(0)
                    pooled_np = pooled.detach().cpu().numpy().tolist()
                    window_vectors.append(pooled_np)
            
            # Set the values on the response dict
            r['text'] = text
            r['reason'] = text  # preserve reason to prevent breaking legacy code
            if 'label' in agent:
                r['label'] = agent['label']
                
            r['st_embedding'] = st_embed
            
            r['tk_embedding'] = token_embeddings
            
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
            
            
            