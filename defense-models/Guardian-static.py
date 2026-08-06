"""GUARDIAN static (single-snapshot) attributed-graph anomaly detector.

Implements the static DOMINANT-style detector from the public GUARDIAN
repository (``model_static.py``) adapted to GAMMAF's embedding contract:
each debate round is passed through a GNN autoencoder that reconstructs node
attributes and graph structure from that snapshot alone.

Unlike the temporal Guardian (``Guardian.py``) there is no VAE
reparameterization, no GIB penalty and no temporal aggregation: node states
are deterministic encoder outputs and anomaly scores are the reconstruction
errors of the current round only.  The original BERT text encoder is replaced
by GAMMAF's precomputed ``st_embedding`` vectors.
"""

from __future__ import annotations

import argparse
import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from LoggingUtils import log_done, log_info, log_warn, print_epoch_log
from Utils import load_config_from_path


def _adjacency(adj: Any, n: int) -> np.ndarray:
    if adj is None:
        return np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)
    a = np.asarray(adj, dtype=np.float32)
    if a.shape != (n, n):
        raise ValueError(f"GuardianStatic adjacency must have shape {(n, n)}, got {a.shape}")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("GuardianStatic adjacency must contain finite non-negative values")
    return a


def _normalized_adjacency(adj: torch.Tensor) -> torch.Tensor:
    n = adj.shape[-1]
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    a = adj + eye
    degree = a.sum(-1).clamp_min(1e-12)
    inv = degree.rsqrt()
    return inv.unsqueeze(-1) * a * inv.unsqueeze(-2)


class DOMINANTNet(nn.Module):
    """Single-snapshot GNN autoencoder (static DOMINANT).

    A multi-layer GCN encodes the current round's node features and two
    decoders reconstruct node attributes and the binary adjacency matrix.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_gnn_layers: int, dropout: float):
        super().__init__()
        if num_gnn_layers < 1:
            raise ValueError("GuardianStatic num_gnn_layers must be >= 1")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        layers = [nn.Linear(input_dim, hidden_dim)]
        layers.extend(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gnn_layers - 1))
        self.gcn_layers = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(dropout)
        self.attr_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)
        )
        # The reference model scores every ordered node pair with the inner
        # product of a structure projection rather than an MLP edge decoder.
        self.structure_decoder = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [agents, features], adj: [agents, agents]
        norm = _normalized_adjacency(adj)
        h = x
        for layer in self.gcn_layers:
            h = F.relu(layer(norm @ h))
            h = self.dropout_layer(h)
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        z = self.encode(x, adj)
        x_hat = self.attr_decoder(z)
        s = self.structure_decoder(z)
        adjacency_reconstructed = torch.sigmoid(s @ s.transpose(-1, -2))
        return x_hat, adjacency_reconstructed, z


class _RoundDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x, a = self.samples[index]
        return torch.from_numpy(x).float(), torch.from_numpy(a).float()


class GuardianStatic:
    """Training and inference object consumed by the evaluation framework."""

    def __init__(self, args):
        self.config = args
        self.device = torch.device(getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: DOMINANTNet | None = None
        self._predict_lock = threading.RLock()
        self._validate_config()

    def _validate_config(self):
        self.hidden_dim = int(getattr(self.config, "hidden_dim", 128))
        self.num_gnn_layers = int(getattr(self.config, "num_gnn_layers", 2))
        self.dropout = float(getattr(self.config, "dropout", 0.0))
        # The static implementation follows the reference DOMINANT detector,
        # which uses feature_weight=0.3 and structure_weight=0.7.
        self.alpha = float(getattr(self.config, "alpha", 0.3))
        self.learning_rate = float(getattr(self.config, "learning_rate", 1e-3))
        self.weight_decay = float(getattr(self.config, "weight_decay", 1e-4))
        self.epochs = int(getattr(self.config, "num_epochs", getattr(self.config, "epochs", 20)))
        self.batch_size = int(getattr(self.config, "batch_size", 16))
        self.val_split = float(getattr(self.config, "val_split", 0.2))
        self.lr_patience = int(getattr(self.config, "lr_patience", getattr(self.config, "lr_patience_max", 5)))
        self.lr_factor = float(getattr(
            self.config,
            "lr_factor",
            getattr(self.config, "lr_patience_factor", getattr(self.config, "lr_reduce_factor", 0.5)),
        ))
        self.early_stop = int(getattr(self.config, "early_stop", 10))
        self.remove_count = int(getattr(self.config, "remove_count", getattr(self.config, "top_k", 1)))
        if self.num_gnn_layers < 1:
            raise ValueError("GuardianStatic num_gnn_layers must be >= 1")
        if not 0 <= self.alpha <= 1:
            raise ValueError("GuardianStatic alpha must be in [0, 1]")
        if not 0 <= self.val_split < 1:
            raise ValueError("GuardianStatic val_split must be in [0, 1)")
        if self.epochs < 1 or self.batch_size < 1 or self.remove_count < 0:
            raise ValueError("GuardianStatic epochs, batch_size, and remove_count are invalid")
        if self.lr_patience < 1 or not 0 < self.lr_factor < 1 or self.early_stop < 1:
            raise ValueError("GuardianStatic lr_patience and early_stop must be >= 1, and lr_factor must be in (0, 1)")

    @staticmethod
    def _round_embedding(round_data) -> np.ndarray:
        if isinstance(round_data, np.ndarray):
            x = round_data
        else:
            try:
                x = np.asarray([item["st_embedding"] for item in round_data], dtype=np.float32)
            except (KeyError, TypeError) as exc:
                raise ValueError("GuardianStatic round must contain st_embedding for every agent") from exc
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or not np.isfinite(x).all():
            raise ValueError(f"GuardianStatic round embeddings must be finite 2-D data, got {x.shape}")
        return x.astype(np.float32, copy=False)

    def _make_model(self, input_dim):
        if getattr(self.config, "input_dim", None) is not None and int(self.config.input_dim) != input_dim:
            raise ValueError(f"GuardianStatic input_dim={self.config.input_dim} does not match embeddings ({input_dim})")
        self.model = DOMINANTNet(input_dim, self.hidden_dim, self.num_gnn_layers, self.dropout).to(self.device)

    def _samples_from_pickle(self, path):
        with open(path, "rb") as handle:
            raw = pickle.load(handle)
        records = raw.get("data", raw) if isinstance(raw, dict) else raw
        samples = []
        group_ids = []
        debate_id = 0
        for entry in records:
            if not isinstance(entry, dict):
                continue
            debates = entry.get("results", [entry])
            base_adj = entry.get("adj_matrix", entry.get("topology", entry.get("adjacency_matrix")))
            for debate in debates:
                if not isinstance(debate, dict) or "debate_rounds" not in debate:
                    continue
                rounds = [self._round_embedding(r) for r in debate["debate_rounds"]]
                if not rounds:
                    continue
                n, d = rounds[0].shape
                if any(r.shape != (n, d) for r in rounds):
                    raise ValueError("GuardianStatic training debate has inconsistent agent or embedding dimensions")
                a = _adjacency(debate.get("adj_matrix", debate.get("topology", base_adj)), n)
                for r in rounds:
                    samples.append((r, a))
                    group_ids.append(debate_id)
                debate_id += 1
        if not samples:
            raise ValueError(f"No valid debates found in GuardianStatic training data: {path}")
        # Keep group membership alongside the flattened samples. This lets
        # train() split complete debates instead of correlated rounds.
        self._sample_group_ids = group_ids
        self._sample_group_source = samples
        return samples

    def _sample_loss(self, sample_x, sample_a):
        x_hat, adjacency_reconstructed, _ = self.model(sample_x, sample_a)
        target_a = (sample_a > 0).float()
        attr = F.mse_loss(x_hat, sample_x)
        structure = F.binary_cross_entropy(adjacency_reconstructed, target_a)
        return self.alpha * attr + (1 - self.alpha) * structure

    def _evaluate(self, loader):
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                for sample_x, sample_a in batch:
                    total += self._sample_loss(sample_x.to(self.device), sample_a.to(self.device)).item()
                    count += 1
        return total / max(1, count)

    def train(self, topology_data):
        """Train on normalized round samples or the framework's raw topology record."""
        samples = topology_data
        if isinstance(topology_data, dict) and "windows" in topology_data:
            samples = topology_data["windows"]
        grouped_samples = samples is getattr(self, "_sample_group_source", None)
        group_ids = getattr(self, "_sample_group_ids", None) if grouped_samples else None
        samples = [(np.asarray(x, dtype=np.float32), _adjacency(a, np.asarray(x).shape[0])) for x, a in samples]
        input_dim = samples[0][0].shape[-1]
        if any(x.ndim != 2 or x.shape[-1] != input_dim for x, _ in samples):
            raise ValueError("GuardianStatic training samples must have shape [agents, embedding_dim]")
        self._make_model(input_dim)

        split_seed = int(getattr(self.config, "split_seed", getattr(self.config, "seed", 0)))
        split_rng = np.random.default_rng(split_seed)
        unique_groups = np.unique(group_ids) if group_ids is not None and len(group_ids) == len(samples) else []
        if len(unique_groups) >= 2 and self.val_split > 0:
            group_order = split_rng.permutation(unique_groups)
            val_group_count = max(1, int(round(len(unique_groups) * self.val_split)))
            val_group_count = min(val_group_count, len(unique_groups) - 1)
            val_groups = set(group_order[:val_group_count].tolist())
            val_indices = [i for i, group in enumerate(group_ids) if group in val_groups]
            train_indices = [i for i, group in enumerate(group_ids) if group not in val_groups]
            val_samples = [samples[i] for i in val_indices]
            train_samples = [samples[i] for i in train_indices]
            log_info(
                f"GuardianStatic validation split uses {val_group_count} of {len(unique_groups)} debates "
                f"({len(val_samples)} validation samples; {len(train_samples)} training samples)."
            )
        elif len(samples) >= 2 and self.val_split > 0:
            # Preserve support for callers that provide already-flattened
            # samples without debate identifiers.
            indices = split_rng.permutation(len(samples))
            val_count = max(1, int(round(len(samples) * self.val_split)))
            val_count = min(val_count, len(samples) - 1)
            val_samples = [samples[i] for i in indices[:val_count]]
            train_samples = [samples[i] for i in indices[val_count:]]
            log_warn("GuardianStatic debate-group split is unavailable; splitting individual round samples")
        else:
            log_warn("GuardianStatic validation split is unavailable; validating on the training samples")
            train_samples = samples
            val_samples = samples

        dataloader_seed = int(getattr(self.config, "dataloader_seed", getattr(self.config, "seed", 0)))
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(dataloader_seed)
        loader = DataLoader(_RoundDataset(train_samples), batch_size=self.batch_size, shuffle=True,
                            generator=dataloader_generator, collate_fn=lambda batch: batch)
        val_loader = DataLoader(_RoundDataset(val_samples), batch_size=self.batch_size, shuffle=False,
                                collate_fn=lambda batch: batch)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        lr_wait = 0
        early_wait = 0
        training_info = {"best_epoch": None, "best_val_loss": None, "stopped_epoch": None, "lr_reductions": 0}
        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for batch in loader:
                optimizer.zero_grad()
                losses = []
                for sample_x, sample_a in batch:
                    sample_x, sample_a = sample_x.to(self.device), sample_a.to(self.device)
                    losses.append(self._sample_loss(sample_x, sample_a))
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total += loss.item()

            train_loss = total / max(1, len(loader))
            val_loss = self._evaluate(val_loader)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                lr_wait = 0
                early_wait = 0
                log_info(f"[BEST] Validation improved at epoch {epoch + 1}; saving model state.")
            else:
                stop_after_epoch = False
                lr_wait += 1
                early_wait += 1

                if lr_wait >= self.lr_patience:
                    old_lr = optimizer.param_groups[0]["lr"]
                    new_lr = old_lr * self.lr_factor
                    for group in optimizer.param_groups:
                        group["lr"] = new_lr
                    lr_wait = 0
                    training_info["lr_reductions"] += 1
                    log_info(
                        f"No validation improvement for {self.lr_patience} epochs. "
                        f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}"
                    )

                if early_wait >= self.early_stop:
                    training_info["stopped_epoch"] = epoch + 1
                    stop_after_epoch = True

            current_lr = optimizer.param_groups[0]["lr"]
            print_epoch_log(epoch + 1, self.epochs, train_loss, val_loss, current_lr, is_best)
            if not is_best and stop_after_epoch:
                log_info(f"Early stopping triggered after {self.early_stop} epochs without validation improvement.")
                break

        if best_model_state is None:
            raise RuntimeError("GuardianStatic training completed without a validation model state")
        self.best_model_state = {k: v.clone() for k, v in best_model_state.items()}
        self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
        training_info["best_epoch"] = best_epoch
        training_info["best_val_loss"] = best_val_loss
        self.training_info = training_info
        if training_info["stopped_epoch"] is not None:
            log_info(f"Restoring best model from epoch {best_epoch}.")
        log_done(f"GuardianStatic training complete. Best validation loss: {best_val_loss:.6f}")

    def predict(self, round_data, adj_matrix, temporal_rounds=None):
        """Score a single round; ``temporal_rounds`` is accepted for interface
        compatibility with the temporal Guardian adapter and is ignored."""
        if self.model is None:
            raise RuntimeError("GuardianStatic model is not initialized; train or load a checkpoint first")
        x_np = self._round_embedding(round_data)
        default_adj = _adjacency(None, x_np.shape[0])
        adj = _adjacency(adj_matrix if adj_matrix is not None else default_adj, x_np.shape[0])
        with self._predict_lock, torch.no_grad():
            self.model.eval()
            x = torch.from_numpy(x_np).float().to(self.device)
            a = torch.from_numpy(adj).float().to(self.device)
            x_hat, adjacency_reconstructed, _ = self.model(x, a)
            attr_error = ((x_hat - x) ** 2).mean(dim=1)
            edge_error = ((adjacency_reconstructed - (a > 0).float()) ** 2).mean(dim=1)
            scores = (self.alpha * attr_error + (1 - self.alpha) * edge_error).cpu().numpy().astype(float)
        threshold = getattr(self.config, "threshold", None)
        flags = np.zeros(len(scores), dtype=int)
        if threshold is not None:
            flags[scores > float(threshold)] = 1
        elif self.remove_count:
            flags[np.argsort(-scores)[:min(self.remove_count, len(scores))]] = 1
        return flags, scores

    def reset(self):
        """Clear inference-only state; the static detector keeps no history."""
        return None

    def save_model(self, path):
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized GuardianStatic model")
        torch.save({"model_state_dict": self.model.state_dict(), "input_dim": self.model.input_dim,
                    "hidden_dim": self.hidden_dim, "num_gnn_layers": self.num_gnn_layers,
                    "dropout": self.dropout}, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        required = {"model_state_dict", "input_dim", "hidden_dim", "num_gnn_layers", "dropout"}
        if not required.issubset(checkpoint):
            raise ValueError(f"GuardianStatic checkpoint is missing keys: {sorted(required - set(checkpoint))}")
        self.model = DOMINANTNet(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["num_gnn_layers"],
                                 checkpoint["dropout"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


class Master:
    def __init__(self, config_path):
        self.args = load_config_from_path(config_path)

    def _run(self, train_pkl_path=None):
        path = train_pkl_path or getattr(self.args, "pkl_train", None)
        if not path:
            raise ValueError("GuardianStatic requires pkl_train or a training path")
        model = GuardianStatic(self.args)
        samples = model._samples_from_pickle(path)
        model.train(samples)
        return {}, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    GuardianStatic(load_config_from_path(args.config)).train(
        GuardianStatic(load_config_from_path(args.config))._samples_from_pickle(load_config_from_path(args.config).pkl_train)
    )
