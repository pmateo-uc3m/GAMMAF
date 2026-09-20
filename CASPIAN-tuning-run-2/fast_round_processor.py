"""Lightweight round processor for the CASPIAN runs.

CASPIAN consumes only the pooled sentence embedding ``st_embedding``; it never
reads ``tk_embedding``.  The stock ``RoundProcessor`` computes both, which
doubles the encoder work (the live-evaluation loop constructs the processor
with CPU because it cannot manage many concurrent GPU calls).  This class:

* computes ``st_embedding`` with the *exact same* chunking and token-weighted
  averaging as ``RoundProcessor`` (verified numerically), so the CASPIAN input
  is unchanged;
* omits the token-level encoder (unused by CASPIAN);
* transparently uses the GPU when one is available, serialising encode calls
  with a lock so the single shared model is safe under the framework's thread
  pool.  The embedding model is ~90 MB; the vLLM server leaves >12 GB free.

It is referenced from the CASPIAN run configs via ``text_processor_path`` /
``text_processor_class_name``; no existing file is modified.
"""

import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_MODEL_INIT_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()


class FastRoundProcessor:
    def __init__(self, device='cpu'):
        # The framework passes device='cpu' for concurrency safety; use the GPU
        # when available because encode calls are serialised by _ENCODE_LOCK.
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.chunk_overlap = 32
        self._model = None

    def _get_model(self):
        if self._model is None:
            with _MODEL_INIT_LOCK:
                if self._model is None:
                    self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def _encode_text(self, text, st_model):
        max_length = int(getattr(st_model, 'max_seq_length', 256) or 256)
        special_tokens = st_model.tokenizer.num_special_tokens_to_add(pair=False)
        chunk_size = max_length - special_tokens
        if chunk_size < 1:
            raise ValueError("Text encoder maximum length cannot fit special tokens")
        overlap = min(self.chunk_overlap, chunk_size - 1)

        token_ids = st_model.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )["input_ids"]
        if not token_ids:
            token_chunks = [[]]
        else:
            step = chunk_size - overlap
            token_chunks = [
                token_ids[start:start + chunk_size]
                for start in range(0, len(token_ids), step)
            ]

        chunks = [
            st_model.tokenizer.decode(ids, skip_special_tokens=True) or " "
            for ids in token_chunks
        ]
        chunk_weights = np.asarray(
            [max(1, len(ids)) for ids in token_chunks], dtype=np.float32
        )

        with _ENCODE_LOCK:
            chunk_embeddings = np.asarray(
                st_model.encode(
                    chunks,
                    device=self.device,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        if chunk_embeddings.ndim == 1:
            chunk_embeddings = chunk_embeddings[None, :]
        return np.average(chunk_embeddings, axis=0, weights=chunk_weights)

    def process_round(self, round_data):
        """Transform each agent's message into a pooled sentence embedding."""
        st_model = self._get_model()
        embedded_round = []
        for agent in round_data:
            processed = {key: agent[key] for key in agent if key != 'message'}
            processed['st_embedding'] = self._encode_text(agent['message'], st_model)
            embedded_round.append(processed)
        return embedded_round
