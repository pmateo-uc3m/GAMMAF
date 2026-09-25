"""DOMINANT: Deep Anomaly Detection on Attributed Networks.

Paper: Ding, Li, Bhanushali & Liu, "Deep Anomaly Detection on Attributed
Networks" (SIAM SDM 2019).

DOMINANT is a graph autoencoder that reconstructs BOTH the attribute matrix and
the adjacency matrix of an attributed graph, and scores each node by its
weighted reconstruction error. The mapping to this codebase is as follows:

* Each debate round is an attributed-graph snapshot: nodes are agents, node
  features are the sentence embeddings (``st_embedding``), edges are the
  adjacency matrix.
* GCN layer (Eq. 3):  H' = ReLU( A~ @ (H W) + b ),  with the symmetrically
  normalized adjacency A~ = D~^-1/2 (A + I) D~^-1/2.
* Encoder (2-layer GCN) produces node embeddings Z.
* Attribute decoder (2-layer GCN) reconstructs X_hat from Z.
* Structure decoder (1-layer GCN + inner product) reconstructs the adjacency
  A_hat = Z_s Z_s^T.
* Anomaly score (Eq. 11):
      score(v_i) = alpha * ||x_i - x_i_hat||_2 + (1 - alpha) * ||a_i - a_i_hat||_2
  where a_i is row i of A + I. Training minimizes the mean of these scores.

The architecture, layer sizes and loss follow the authors' official PyTorch
implementation (GCN_AnomalyDetection_pytorch). The original BERT/TF-IDF
features are replaced by GAMMAF's sentence embeddings, matching the rest of the
project.
"""

from __future__ import annotations

import argparse
import math
import random
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import sys

_DIR = Path(__file__).resolve().parent
sys.path.append(str(_DIR.parent))
sys.path.insert(0, str(_DIR))

from LoggingUtils import log_done, log_info, log_section, log_warn, print_epoch_log, LRPlateauReducer
from EvaluationConfigCheck import load_defense_model_config

# Reuse BlindGuard's training-data loader so DOMINANT consumes the exact same
# sentence-embedding / graph representation (no data perturbation is applied).
from BlindGuard import TrainDataProcessor


class GraphConvolution(nn.Module):
    """Simple GCN layer: output = adj_norm @ (input @ W) + b."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj_norm):
        support = x @ self.weight
        output = adj_norm @ support
        return output + self.bias


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        x = F.relu(self.gc1(x, adj_norm))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj_norm))
        return x


class AttributeDecoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        x = F.relu(self.gc1(x, adj_norm))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj_norm))
        return x


class StructureDecoder(nn.Module):
    def __init__(self, nhid, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj_norm):
        x = F.relu(self.gc1(x, adj_norm))
        x = F.dropout(x, self.dropout, training=self.training)
        return x @ x.transpose(-1, -2)


class DOMINANTNet(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super().__init__()
        self.input_dim = feat_size
        self.hidden_dim = hidden_size
        self.dropout = dropout
        self.encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = AttributeDecoder(feat_size, hidden_size, dropout)
        self.struct_decoder = StructureDecoder(hidden_size, dropout)

    def forward(self, x, adj_norm):
        z = self.encoder(x, adj_norm)
        x_hat = self.attr_decoder(z, adj_norm)
        struct_reconstructed = self.struct_decoder(z, adj_norm)
        return struct_reconstructed, x_hat


def _sym_normalize(adj):
    """Symmetrically normalized adjacency with self-loops: D~^-1/2 (A+I) D~^-1/2."""
    n = adj.shape[0]
    a_tilde = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    degree = a_tilde.sum(dim=1).clamp_min(1e-12)
    inv = torch.pow(degree, -0.5)
    return inv.unsqueeze(1) * a_tilde * inv.unsqueeze(0)


def _adj_label(adj):
    """Structure reconstruction target: A + I (self-loops included)."""
    n = adj.shape[0]
    return adj + torch.eye(n, device=adj.device, dtype=adj.dtype)


def dominant_node_scores(x, adj_label, adj_norm, model, alpha):
    """Per-node anomaly scores (Eq. 11) plus the two component costs."""
    struct_reconstructed, x_hat = model(x, adj_norm)
    attr_error = torch.sqrt(((x_hat - x) ** 2).sum(dim=1))
    struct_error = torch.sqrt(((struct_reconstructed - adj_label) ** 2).sum(dim=1))
    scores = alpha * attr_error + (1 - alpha) * struct_error
    return scores, struct_error.mean(), attr_error.mean()


def build_dominant_samples(train_data):
    """Return a list of (x, adj) samples, one per debate round."""
    samples = []
    for topology in train_data.data:
        adj = np.asarray(topology["adj_matrix"], dtype=np.float32)
        for debate in topology["debates"]:
            for round_idx in range(debate.shape[0]):
                x = np.asarray(debate[round_idx], dtype=np.float32)
                samples.append((x, adj))
    return samples


class _DominantDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, a = self.samples[index]
        return torch.from_numpy(x).float(), torch.from_numpy(a).float()


class DOMINANTLoop:
    """Training and inference object consumed by the evaluation framework."""

    def __init__(self, args):
        self.args = args
        self.config = args
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._predict_lock = threading.Lock()
        self._validate_config()

    def _validate_config(self):
        self.hidden_dim = int(self.config.hidden_dim)
        self.dropout = float(self.config.dropout)
        self.alpha = float(self.config.alpha)
        self.learning_rate = float(self.config.learning_rate)
        self.weight_decay = float(self.config.weight_decay)
        self.epochs = int(self.config.epochs)
        self.batch_size = int(self.config.batch_size)
        self.val_split = float(self.config.val_split)
        if not 0 <= self.alpha <= 1:
            raise ValueError("DOMINANT alpha must be in [0, 1]")
        if not 0 <= self.val_split < 1:
            raise ValueError("DOMINANT val_split must be in [0, 1)")
        if self.hidden_dim < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("DOMINANT hidden_dim, epochs, and batch_size must be >= 1")

    @staticmethod
    def _round_embedding(round_data) -> np.ndarray:
        if isinstance(round_data, np.ndarray):
            x = round_data
        else:
            try:
                x = np.asarray([item["st_embedding"] for item in round_data], dtype=np.float32)
            except (KeyError, TypeError) as exc:
                raise ValueError("DOMINANT round must contain st_embedding for every agent") from exc
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or not np.isfinite(x).all():
            raise ValueError(f"DOMINANT round embeddings must be finite 2-D data, got {x.shape}")
        return x.astype(np.float32, copy=False)

    def _make_model(self, input_dim):
        if self.config.input_dim is not None and int(self.config.input_dim) != input_dim:
            raise ValueError(f"DOMINANT input_dim={self.config.input_dim} does not match embeddings ({input_dim})")
        self.model = DOMINANTNet(input_dim, self.hidden_dim, self.dropout).to(self.device)

    def _sample_loss(self, x, adj):
        adj_label = _adj_label(adj)
        adj_norm = _sym_normalize(adj)
        scores, _, _ = dominant_node_scores(x, adj_label, adj_norm, self.model, self.alpha)
        return scores.mean()

    def _evaluate(self, loader):
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                for x, a in batch:
                    total += self._sample_loss(x.to(self.device), a.to(self.device)).item()
                    count += 1
        return total / max(1, count)

    def train(self, samples):
        log_section("Training Phase - DOMINANT")
        samples = [(np.asarray(x, dtype=np.float32), np.asarray(a, dtype=np.float32)) for x, a in samples]
        if not samples:
            raise ValueError("DOMINANT received no training samples.")
        input_dim = samples[0][0].shape[-1]
        if any(x.ndim != 2 or x.shape[-1] != input_dim for x, _ in samples):
            raise ValueError("DOMINANT training samples must have shape [agents, embedding_dim]")
        self._make_model(input_dim)

        split_seed = int(self.config.split_seed)
        n_total = len(samples)
        perm_idx = np.random.default_rng(split_seed).permutation(n_total)
        if n_total >= 2 and 0 < self.val_split < 1:
            n_val = max(1, int(round(n_total * self.val_split)))
            n_val = min(n_val, n_total - 1)
            val_idx = perm_idx[:n_val]
            train_idx = perm_idx[n_val:]
        else:
            train_idx = np.arange(n_total)
            val_idx = np.array([], dtype=int)

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        dataloader_seed = int(self.config.dataloader_seed)
        gen = torch.Generator()
        gen.manual_seed(dataloader_seed)
        loader = DataLoader(_DominantDataset(train_samples), batch_size=self.batch_size, shuffle=True,
                            generator=gen, collate_fn=lambda batch: batch)
        val_loader = DataLoader(_DominantDataset(val_samples), batch_size=self.batch_size, shuffle=False,
                                collate_fn=lambda batch: batch)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_reducer = LRPlateauReducer(
            optimizer,
            patience=self.config.n_epochs_lr_reduce,
            factor=self.config.lr_reduce_factor,
            min_lr=self.config.min_lr,
            min_improvement_pct=self.config.lr_reduce_improvement_pct,
            early_stop_patience=self.config.n_epochs_early_stop,
            early_stop_improvement_pct=self.config.early_stop_improvement_pct,
        )
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            n_seen = 0
            for batch in loader:
                optimizer.zero_grad()
                losses = []
                for x, a in batch:
                    x, a = x.to(self.device), a.to(self.device)
                    losses.append(self._sample_loss(x, a))
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(losses)
                n_seen += len(losses)
            train_loss = total / max(1, n_seen)

            val_loss = self._evaluate(val_loader)
            is_best, reduced, should_stop = lr_reducer.step(val_loss)
            if is_best:
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            print_epoch_log(epoch + 1, self.epochs, train_loss, val_loss, lr_reducer.lr, is_best)
            if should_stop:
                log_warn(f"Early stopping triggered at epoch {epoch + 1} (no improvement for {self.config.n_epochs_early_stop} epochs)")
                break

        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            log_done(f"Training complete. Best model restored with validation loss: {lr_reducer.best_loss:.6f}")
        else:
            log_warn("Training complete. No validation improvement snapshot was captured.")

    def predict(self, round_data, adj_matrix):
        """Score a single debate round; returns (flags, anomaly_scores)."""
        if self.model is None:
            raise RuntimeError("DOMINANT model is not initialized; train or load a checkpoint first")
        x_np = self._round_embedding(round_data)
        adj = np.asarray(adj_matrix, dtype=np.float32)
        if adj.shape != (x_np.shape[0], x_np.shape[0]):
            raise ValueError(f"DOMINANT adjacency must have shape {(x_np.shape[0], x_np.shape[0])}, got {adj.shape}")

        with self._predict_lock, torch.no_grad():
            self.model.eval()
            x = torch.from_numpy(x_np).float().to(self.device)
            a = torch.from_numpy(adj).float().to(self.device)
            adj_label = _adj_label(a)
            adj_norm = _sym_normalize(a)
            scores, _, _ = dominant_node_scores(x, adj_label, adj_norm, self.model, self.alpha)
            scores = scores.cpu().numpy().astype(float)

        n_agents = x_np.shape[0]
        threshold = self.config.threshold
        flags = np.zeros(n_agents, dtype=int)
        if threshold is not None:
            flags[scores > float(threshold)] = 1
        else:
            top_k = int(self.config.top_k)
            flags[np.argsort(-scores)[:min(top_k, n_agents)]] = 1
        return flags, scores

    def save_model(self, path):
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized DOMINANT model")
        torch.save({"model_state_dict": self.model.state_dict(),
                    "input_dim": self.model.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "dropout": self.dropout}, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        required = {"model_state_dict", "input_dim", "hidden_dim", "dropout"}
        if not required.issubset(checkpoint):
            raise ValueError(f"DOMINANT checkpoint is missing keys: {sorted(required - set(checkpoint))}")
        self.model = DOMINANTNet(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


class Master:
    def __init__(self, config_path):
        self.args = load_defense_model_config(config_path)

    def _run(self, train_pkl_path=None):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

        log_info("Loading and processing training data...")
        train_data = TrainDataProcessor({}, target_topologies=self.args.topologies)
        train_data.load_pkl(train_pkl_path or self.args.pkl_train)

        samples = build_dominant_samples(train_data)
        log_info(f"DOMINANT data prepared: {len(samples)} round samples "
                 f"(feature dim {samples[0][0].shape[1]}).")

        log_info("Starting DOMINANT training...")
        loop = DOMINANTLoop(self.args)
        loop.train(samples)
        log_done("DOMINANT model trained.")
        return {}, loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DOMINANT: Deep Anomaly Detection on Attributed Networks"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parsed = parser.parse_args()
    if not parsed.config:
        raise ValueError("--config <path_to_yaml> is required.")
    master = Master(parsed.config)
    _, model = master._run()
