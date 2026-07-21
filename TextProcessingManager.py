from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import threading

class RoundProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self._local = threading.local()

    def _get_local_resources(self):
        if not hasattr(self._local, '_init'):
            self._local.st_model = SentenceTransformer(self.model_id, device=self.device)
            self._local.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._local.hf_model = AutoModel.from_pretrained(self.model_id)
            self._local.hf_model.eval()
            self._local._init = True
        return self._local.st_model, self._local.hf_tokenizer, self._local.hf_model

    def process_round(self, round_data):
        """Transform the 'reason' field of each agent in the round into embeddings."""
        st_model, hf_tokenizer, hf_model = self._get_local_resources()
        embedded_round = []
        for agent in round_data:
            r = {
                key: agent[key] for key in agent if key != 'reason'
            }
            text = agent['reason']

            st_embed = st_model.encode(text, device=self.device)

            # Now the per-token embeddings
            inputs = hf_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = hf_model(**inputs)  # hf_model stays on CPU

            # last_hidden_state: [1, num_tokens, 384]
            token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
            token_embeddings = token_embeddings.tolist()

            r['st_embedding'] = st_embed
            r['tk_embedding'] = token_embeddings

            embedded_round.append(r)
        return embedded_round
            
            
            