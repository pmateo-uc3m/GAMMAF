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
            
            
            