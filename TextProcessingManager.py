from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import threading

class RoundProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.chunk_overlap = 32
        self._local = threading.local()

    def _get_local_resources(self):
        if not hasattr(self._local, '_init'):
            self._local.st_model = SentenceTransformer(self.model_id, device=self.device)
            self._local.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._local.hf_model = AutoModel.from_pretrained(self.model_id)
            self._local.hf_model.eval()
            self._local._init = True
        return self._local.st_model, self._local.hf_tokenizer, self._local.hf_model

    @staticmethod
    def _model_max_length(tokenizer, model, sentence_model):
        limits = [getattr(sentence_model, 'max_seq_length', None)]
        limits.append(getattr(model.config, 'max_position_embeddings', None))
        tokenizer_limit = getattr(tokenizer, 'model_max_length', None)
        if tokenizer_limit is not None and tokenizer_limit < 100000:
            limits.append(tokenizer_limit)
        limits = [int(limit) for limit in limits if limit is not None and int(limit) > 0]
        if not limits:
            raise ValueError("Could not determine the text encoder maximum sequence length")
        return min(limits)

    def _encode_text(self, text, st_model, tokenizer, model):
        max_length = self._model_max_length(tokenizer, model, st_model)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        chunk_size = max_length - special_tokens
        if chunk_size < 1:
            raise ValueError(f"Text encoder maximum length {max_length} cannot fit special tokens")
        overlap = min(self.chunk_overlap, chunk_size - 1)

        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        if not token_ids:
            token_chunks = [[]]
        else:
            step = chunk_size - overlap
            token_chunks = [token_ids[start:start + chunk_size] for start in range(0, len(token_ids), step)]

        chunks = [tokenizer.decode(ids, skip_special_tokens=True) or " " for ids in token_chunks]
        chunk_weights = np.asarray([max(1, len(ids)) for ids in token_chunks], dtype=np.float32)

        chunk_embeddings = np.asarray(st_model.encode(
            chunks,
            device=self.device,
            convert_to_numpy=True,
            show_progress_bar=False,
        ), dtype=np.float32)
        if chunk_embeddings.ndim == 1:
            chunk_embeddings = chunk_embeddings[None, :]
        st_embedding = np.average(chunk_embeddings, axis=0, weights=chunk_weights)

        encoded_chunks = tokenizer(
            chunks,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded_chunks)

        token_embeddings = []
        special_token_ids = []
        for row, mask in zip(encoded_chunks["input_ids"], encoded_chunks["attention_mask"]):
            row_ids = row.tolist()
            valid_length = int(mask.sum().item())
            special_mask = tokenizer.get_special_tokens_mask(
                row_ids[:valid_length], already_has_special_tokens=True
            )
            special_token_ids.append((valid_length, special_mask))

        for index, (valid_length, special_mask) in enumerate(special_token_ids):
            keep = [not is_special for is_special in special_mask]
            token_embeddings.extend(outputs.last_hidden_state[index, :valid_length][keep].cpu().numpy().tolist())

        return st_embedding, token_embeddings

    def process_round(self, round_data):
        """Transform each agent's reason into pooled, non-truncated embeddings."""
        st_model, hf_tokenizer, hf_model = self._get_local_resources()
        embedded_round = []
        for agent in round_data:
            r = {
                key: agent[key] for key in agent if key != 'reason'
            }
            text = agent['reason']

            st_embed, token_embeddings = self._encode_text(text, st_model, hf_tokenizer, hf_model)

            r['st_embedding'] = st_embed
            r['tk_embedding'] = token_embeddings

            embedded_round.append(r)
        return embedded_round
