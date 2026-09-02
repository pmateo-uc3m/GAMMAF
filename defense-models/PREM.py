"""PREM: PREprocessing and Matching for node-level graph anomaly detection.

Paper: Pan et al., "PREM: A Simple Yet Effective Approach for Node-Level Graph
Anomaly Detection" (IEEE ICDM 2023, arXiv:2310.11676).

PREM replaces train-time message passing with a one-shot pre-processing module
and a lightweight ego-neighbor matching network, then trains with a simple
contrastive objective. The mapping to this codebase is as follows:

* Pre-processing module (no trainable weights):
    - ego features   X^(e) = X                       (raw sentence embeddings)
    - neighbor feats X^(n) = P X,  P = M * (D~^-1/2 A~ D~^-1/2)^k   (Eq. 1-2)
      where A~ = A + I (self-loop), D~ = degree(A~), M is the self-anonymization
      mask (zero diagonal, one elsewhere), and k = prop_steps.
* Ego-neighbor matching network (Eq. 3-4):
    - h_i^(e) = x_i^(e) W1 + b1   (fc_ego)
    - h_i^(n) = x_i^(n) W2 + b2   (fc_neighbor)
    - c_i     = cos(h_i^(e), h_i^(n))
* Contrastive training (Eq. 5-8): BCE-like loss over a positive pair
  (ego_i, neighbor_i), a neighbor-based negative (ego_i, neighbor_j) and an
  ego-based negative (ego_i, ego_j), with trade-off weights alpha / gamma.
* Anomaly score (Sec. IV-D): s_i = -c_i^(pos).

The graph structure is the debate adjacency matrix and the node features are the
sentence embeddings produced by the same pipeline used by BlindGuard
(`st_embedding`), so PREM consumes the identical training data format.
"""

from __future__ import annotations

import argparse
import random
import threading
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import sys

_PREM_DIR = Path(__file__).resolve().parent
sys.path.append(str(_PREM_DIR.parent))
sys.path.insert(0, str(_PREM_DIR))

from LoggingUtils import log_done, log_info, log_section, log_warn, print_epoch_log
from Utils import load_config, load_config_from_path

# Reuse BlindGuard's training-data loader so PREM consumes the exact same
# sentence-embedding / graph representation (no data perturbation is applied).
from BlindGuard import TrainDataProcessor


def anonymized_propagation(adj_matrix, k):
    """Anonymized propagation matrix P = M * (D~^-1/2 A~ D~^-1/2)^k (Eq. 2).

    A~ = A + I includes self-loops; D~ is its degree matrix; M is the
    self-anonymization mask whose diagonal is zero and off-diagonal entries are
    one. Zeroing the diagonal removes the ego contribution so X^(n) summarizes
    neighbor information only.
    """
    adj = np.asarray(adj_matrix, dtype=np.float64)
    n = adj.shape[0]
    a_tilde = adj + np.eye(n)
    degree = a_tilde.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    d_inv_sqrt = np.diag(deg_inv_sqrt)
    s = d_inv_sqrt @ a_tilde @ d_inv_sqrt
    s_k = np.linalg.matrix_power(s, int(k))
    np.fill_diagonal(s_k, 0.0)
    return s_k


def build_prem_dataset(train_data, prop_steps, propagation_fn=anonymized_propagation):
    """Turn BlindGuard's processed data into PREM's (ego, neighbor) node pairs.

    For every debate round of every topology the ego features are the raw
    sentence embeddings and the neighbor features are the anonymized propagated
    embeddings (X^(n) = P X). P is precomputed once per topology.
    """
    ego_list = []
    neighbor_list = []
    labels_list = []
    for topology in train_data.data:
        adj_matrix = np.asarray(topology["adj_matrix"])
        prop = propagation_fn(adj_matrix, prop_steps)
        labels = topology.get("labels")
        for debate_idx, debate in enumerate(topology["debates"]):
            agent_labels = labels[debate_idx] if labels is not None else None
            for round_idx in range(debate.shape[0]):
                x = np.asarray(debate[round_idx], dtype=np.float32)
                xn = (prop @ x).astype(np.float32)
                ego_list.append(x)
                neighbor_list.append(xn)
                if agent_labels is not None:
                    labels_list.append(agent_labels)

    ego = np.concatenate(ego_list, axis=0) if ego_list else np.zeros((0, 0), dtype=np.float32)
    neighbor = np.concatenate(neighbor_list, axis=0) if neighbor_list else np.zeros((0, 0), dtype=np.float32)
    labels = np.concatenate(labels_list, axis=0) if labels_list else None

    return {
        "topology_name": "combined",
        "ego": ego,
        "neighbor": neighbor,
        "labels": labels,
    }


class PREMDiscriminator(nn.Module):
    """Ego-neighbor matching network: two linear layers + cosine similarity."""

    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.fc_ego = nn.Linear(input_dim, emb_dim)       # W1, b1  (Eq. 3a)
        self.fc_neighbor = nn.Linear(input_dim, emb_dim)  # W2, b2  (Eq. 3b)

    def forward(self, ego, neighbor):
        """Anomaly score s_i = -cos(h_i^(e), h_i^(n)) (Sec. IV-D)."""
        h_e = self.fc_ego(ego)
        h_n = self.fc_neighbor(neighbor)
        cos = F.cosine_similarity(h_e, h_n, dim=-1, eps=1e-8)
        return -cos


class PREMTopologyLoop:
    """PREM train / inference object consumed by the evaluation framework."""

    def __init__(self, args, propagation_fn=anonymized_propagation,
                 discriminator_cls=PREMDiscriminator):
        self.args = args
        self.config = args
        if getattr(args, "device", None):
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.prop_steps = int(getattr(args, "prop_steps", 2))
        self._propagation_fn = propagation_fn
        self._discriminator_cls = discriminator_cls
        self._predict_lock = threading.Lock()

    def _contrastive_loss(self, ego, neighbor, alpha, gamma):
        """BCE-like contrastive loss (Eq. 8).

        Positive term c_i^(pos)   = cos(h_i^(e), h_i^(n))         (Eq. 5)
        Neighbor negative         = cos(h_i^(e), h_j^(n))         (Eq. 6)
        Ego negative              = cos(h_i^(e), h_k^(e))         (Eq. 7)

        Each c in [-1, 1] is linearly projected into [0, 1] via (c + 1) / 2.
        """
        device = ego.device
        n = ego.shape[0]
        if n < 2:
            perm = torch.arange(n, device=device)
        else:
            perm = torch.randperm(n, device=device)

        h_e = self.model.fc_ego(ego)
        h_n = self.model.fc_neighbor(neighbor)
        h_n_neg = self.model.fc_neighbor(neighbor[perm])  # neighbor-based negative
        h_e_neg = self.model.fc_ego(ego[perm])            # ego-based negative

        def cos(a, b):
            return F.cosine_similarity(a, b, dim=-1, eps=1e-8)

        c_pos = ((cos(h_e, h_n) + 1) / 2).clamp(1e-7, 1 - 1e-7)
        c_neg_nei = ((cos(h_e, h_n_neg) + 1) / 2).clamp(1e-7, 1 - 1e-7)
        c_neg_ego = ((cos(h_e, h_e_neg) + 1) / 2).clamp(1e-7, 1 - 1e-7)

        loss = -(
            c_pos.log().mean()
            + alpha * (1 - c_neg_nei).log().mean()
            + gamma * (1 - c_neg_ego).log().mean()
        )
        return loss

    def train(self, topology_data):
        log_section(f"Training Phase - Topology: {topology_data['topology_name']}")

        ego = np.asarray(topology_data["ego"], dtype=np.float32)
        neighbor = np.asarray(topology_data["neighbor"], dtype=np.float32)
        n_total = ego.shape[0]
        if n_total == 0:
            raise ValueError("PREM received no training samples.")
        if ego.shape != neighbor.shape:
            raise ValueError("PREM ego and neighbor feature matrices must have the same shape.")

        input_dim = ego.shape[1]
        cfg_input_dim = getattr(self.args, "input_dim", None)
        if cfg_input_dim is not None and int(cfg_input_dim) != input_dim:
            raise ValueError(f"PREM input_dim={cfg_input_dim} does not match embeddings ({input_dim}).")
        emb_dim = int(getattr(self.args, "emb_dim", 128))
        alpha = float(getattr(self.args, "alpha", 0.9))
        gamma = float(getattr(self.args, "gamma", 0.1))
        num_epochs = int(getattr(self.args, "num_epochs", getattr(self.args, "epochs", 100)))
        batch_size = int(getattr(self.args, "batch_size", -1))
        learning_rate = float(getattr(self.args, "learning_rate", 1e-3))
        weight_decay = float(getattr(self.args, "weight_decay", 0.0))
        val_split = float(getattr(self.args, "val_split", 0.2))

        # Validation split (indices only; training is fully unsupervised).
        split_seed = int(getattr(self.args, "split_seed", getattr(self.args, "seed", 0)))
        if n_total >= 2 and 0 < val_split < 1:
            perm_idx = np.random.default_rng(split_seed).permutation(n_total)
            n_val = max(1, int(round(n_total * val_split)))
            n_val = min(n_val, n_total - 1)
            val_idx = perm_idx[:n_val]
            train_idx = perm_idx[n_val:]
        else:
            train_idx = np.arange(n_total)
            val_idx = np.array([], dtype=int)

        if batch_size <= 0:
            batch_size = max(1, len(train_idx))

        self.model = self._discriminator_cls(input_dim, emb_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        ego_t = torch.from_numpy(ego)
        neighbor_t = torch.from_numpy(neighbor)

        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            n_seen = 0
            shuffled = np.random.permutation(train_idx)
            for start in range(0, len(shuffled), batch_size):
                bidx = shuffled[start:start + batch_size]
                ego_b = ego_t[bidx].to(self.device)
                neighbor_b = neighbor_t[bidx].to(self.device)
                optimizer.zero_grad()
                loss = self._contrastive_loss(ego_b, neighbor_b, alpha, gamma)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bidx)
                n_seen += len(bidx)
            train_loss = total_loss / max(1, n_seen)

            self.model.eval()
            val_loss = 0.0
            n_val_seen = 0
            with torch.no_grad():
                for start in range(0, len(val_idx), batch_size):
                    bidx = val_idx[start:start + batch_size]
                    ego_b = ego_t[bidx].to(self.device)
                    neighbor_b = neighbor_t[bidx].to(self.device)
                    val_loss += self._contrastive_loss(ego_b, neighbor_b, alpha, gamma).item() * len(bidx)
                    n_val_seen += len(bidx)
            val_loss = val_loss / max(1, n_val_seen)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            current_lr = optimizer.param_groups[0]["lr"]
            print_epoch_log(epoch + 1, num_epochs, train_loss, val_loss, current_lr, is_best)

        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            log_done(f"Training complete. Best model restored with validation loss: {best_val_loss:.6f}")
        else:
            log_warn("Training complete. No validation improvement snapshot was captured.")

    def predict(self, debate_embeddings, adj_matrix):
        """Score a single debate round.

        Args:
            debate_embeddings: list of agent dicts with 'st_embedding', or a
                (n_agents, d) array of node features.
            adj_matrix: adjacency matrix (n_agents x n_agents).

        Returns:
            (flags, anomaly_scores) where flags are 0/1 and scores are higher
            for more anomalous nodes.
        """
        if self.model is None:
            raise RuntimeError("PREM model is not initialized; train or load a checkpoint first.")

        if isinstance(debate_embeddings, np.ndarray):
            x = debate_embeddings
        else:
            x = np.asarray([agent["st_embedding"] for agent in debate_embeddings], dtype=np.float32)
        n_agents = x.shape[0]

        prop = self._propagation_fn(np.asarray(adj_matrix), self.prop_steps)
        xn = (prop @ x).astype(np.float32)

        with self._predict_lock, torch.no_grad():
            self.model.eval()
            x_t = torch.from_numpy(x).float().to(self.device)
            xn_t = torch.from_numpy(xn).float().to(self.device)
            scores = self.model(x_t, xn_t).cpu().numpy().astype(float)

        threshold = getattr(self.config, "threshold", None)
        flags = np.zeros(n_agents, dtype=int)
        if threshold is not None:
            flags[scores > float(threshold)] = 1
        else:
            top_k = int(getattr(self.config, "top_k", 1))
            flags[np.argsort(-scores)[:min(top_k, n_agents)]] = 1

        return flags, scores

    def save_model(self, path):
        if self.model is not None:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.model.input_dim,
                "emb_dim": self.model.emb_dim,
            }
            torch.save(checkpoint, path)
            log_done(f"Model saved to {path}")
        else:
            log_warn("No model to save.")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self._discriminator_cls(checkpoint["input_dim"], checkpoint["emb_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        log_info(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(cls, model_path, device="cuda"):
        args = types.SimpleNamespace(device=device, prop_steps=2)
        instance = cls(args)
        instance.load_model(model_path)
        return instance


class Master:
    def __init__(self, config_path):
        self.args = load_config_from_path(config_path)

    def _run(self, train_pkl_path=None):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

        log_info("Loading and processing training data...")
        train_data = TrainDataProcessor({}, target_topologies=self.args.topologies)
        train_data.load_pkl(train_pkl_path or self.args.pkl_train)

        prop_steps = int(getattr(self.args, "prop_steps", 2))
        prem_data = build_prem_dataset(train_data, prop_steps)
        log_info(f"PREM data prepared: {prem_data['ego'].shape[0]} ego/neighbor node pairs "
                 f"(feature dim {prem_data['ego'].shape[1]}).")

        log_info("Starting PREM training...")
        prem_loop = PREMTopologyLoop(self.args)
        prem_loop.train(prem_data)
        log_done("PREM model trained.")
        return {}, prem_loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PREM: PREprocessing and Matching for Node-Level Graph Anomaly Detection"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parsed = parser.parse_args()
    if not parsed.config:
        raise ValueError("--config <path_to_yaml> is required.")
    master = Master(parsed.config)
    _, model = master._run()
