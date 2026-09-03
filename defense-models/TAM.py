"""TAM: Truncated Affinity Maximization for node-level graph anomaly detection.

Paper: Qiao & Pang, "Truncated Affinity Maximization: One-class Homophily
Modeling for Graph Anomaly Detection" (NeurIPS 2023, arXiv:2306.00006).

TAM leverages the "one-class homophily" property (normal nodes are more similar
to their neighbors than anomalies are). It scores a node by its *local affinity*
to its neighbors in a learned representation space, and learns that space by
maximizing local affinity on iteratively truncated graphs.

Components mapped to this codebase:

* Local node affinity (Eq. 1-2):  h(v_i) = (1/|N(i)|) sum_{j in N(i)} sim(h_i, h_j)
  with cosine similarity; anomaly score f_TAM(v_i) = -h(v_i).
* LAMNet (Eq. 3-4, 9): two-layer GCN (PReLU) producing node embeddings
  h_i = GCN(GCN(X, A~); A~), where A~ is a truncated adjacency.
* NSGT (Eq. 6-8): probabilistically removes edge e_ij when the Euclidean
  attribute distance d_ij exceeds both random thresholds r_i ~ U[d_mean, d_i,max]
  and r_j ~ U[d_mean, d_j,max]; applied sequentially K times.
* Objective (Eq. 5): maximize local affinity (computed on the ORIGINAL adjacency
  A) plus a lambda-weighted regularization that pushes non-neighbor embeddings
  apart.
* Ensemble (Eq. 10): T independent NSGT trees x K truncation depths = T*K LAMNets;
  the anomaly score is 1 - normalized mean affinity over all LAMNets.

Each debate round is treated as an attributed graph (nodes = agents, features =
sentence embeddings, edges = adjacency), matching the rest of the project. The
original node attributes are replaced by GAMMAF's ``st_embedding`` vectors.
"""

from __future__ import annotations

import argparse
import random
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys

_DIR = Path(__file__).resolve().parent
sys.path.append(str(_DIR.parent))
sys.path.insert(0, str(_DIR))

from LoggingUtils import log_done, log_info, log_section, log_warn, print_epoch_log
from Utils import load_config, load_config_from_path

from BlindGuard import TrainDataProcessor


def build_tam_samples(train_data):
    """Return a list of (x, adj) samples, one per debate round (raw adjacency)."""
    samples = []
    for topology in train_data.data:
        adj = np.asarray(topology["adj_matrix"], dtype=np.float32)
        for debate in topology["debates"]:
            for round_idx in range(debate.shape[0]):
                x = np.asarray(debate[round_idx], dtype=np.float32)
                samples.append((x, adj))
    return samples


def _euclidean_dist(x):
    """Pairwise Euclidean distance matrix (n, n)."""
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def nsgt_truncate(adj, x, rng):
    """One NSGT iteration (Eq. 6-8). adj includes self-loops and is unmodified in-place.

    Returns a new truncated adjacency matrix.
    """
    adj = adj.copy()
    n = adj.shape[0]
    dist = _euclidean_dist(x) * adj          # distances on existing edges only
    nonzero = dist[dist != 0]
    mean_dis = float(nonzero.mean()) if nonzero.size > 0 else 0.0

    for i in range(n):
        neigh = np.argwhere(adj[i] > 0).reshape(-1)
        if neigh.size == 0:
            continue
        max_dis = dist[i, neigh].max()
        if max_dis > mean_dis:
            r = rng.uniform(mean_dis, max_dis)  # r_i ~ U[d_mean, d_i,max]
            cutting = neigh[dist[i, neigh] > r]
            if cutting.size > 0:
                adj[i, cutting] = 0

    # Symmetrize (OR): an edge survives unless removed by both endpoints (Eq. 6).
    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj.astype(np.float32)


def _sym_normalize(adj):
    """D^-1/2 A D^-1/2 for a symmetric adjacency (self-loops already present)."""
    n = adj.shape[0]
    degree = adj.sum(dim=1).clamp_min(1e-12)
    inv = torch.pow(degree, -0.5)
    return inv.unsqueeze(1) * adj * inv.unsqueeze(0)


def local_affinity(emb, adj):
    """Per-node local affinity h(v_i) (Eq. 1) on the given (original) adjacency."""
    emb = emb / torch.norm(emb, dim=-1, keepdim=True).clamp_min(1e-12)
    sim = emb @ emb.t()
    sim = sim * adj
    degree = adj.sum(dim=1).clamp_min(1e-12)
    return sim.sum(dim=1) / degree


def reg_loss(emb, adj):
    """Regularization: mean similarity to non-adjacent nodes (Eq. 5 second term)."""
    emb = emb / torch.norm(emb, dim=-1, keepdim=True).clamp_min(1e-12)
    sim = emb @ emb.t()
    adj_inv = 1.0 - adj
    sim = sim * adj_inv
    row_sum = adj_inv.sum(dim=1).clamp_min(1e-12)
    return (sim.sum(dim=1) / row_sum).sum()


class GCNLayer(nn.Module):
    """Single GCN layer: act(D^-1/2 A D^-1/2 (X W) + b), PReLU by default."""

    def __init__(self, in_dim, out_dim, activation="prelu"):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.PReLU() if activation == "prelu" else activation
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        self.bias.data.fill_(0.0)

    def forward(self, x, adj_norm):
        return self.act(adj_norm @ self.fc(x) + self.bias)


class LAMNet(nn.Module):
    """Local Affinity Maximization Network: two-layer GCN (Eq. 3-4)."""

    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.input_dim = in_dim
        self.emb_dim = emb_dim
        self.gcn1 = GCNLayer(in_dim, 2 * emb_dim)
        self.gcn2 = GCNLayer(2 * emb_dim, emb_dim)
        # feat1 is used by the regularization term; feat2 mirrors the reference
        # (it is computed but not used in the affinity-maximization objective).
        self.fc1 = nn.Linear(emb_dim, 2 * emb_dim, bias=False)
        self.fc2 = nn.Linear(emb_dim, 2 * emb_dim, bias=False)

    def forward(self, x, adj_norm):
        h = self.gcn1(x, adj_norm)
        h = self.gcn2(h, adj_norm)
        feat1 = self.fc1(h)
        feat2 = self.fc2(h)
        return h, feat1, feat2


class TAMLoop:
    """Training and inference object consumed by the evaluation framework."""

    def __init__(self, args):
        self.args = args
        self.config = args
        if getattr(args, "device", None):
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []  # list of (tree, cut, LAMNet)
        self._seed = int(getattr(args, "seed", 0))
        self._predict_lock = threading.Lock()
        self._validate_config()

    def _validate_config(self):
        self.emb_dim = int(getattr(self.config, "emb_dim", 128))
        self.num_trees = int(getattr(self.config, "num_trees", 3))      # T
        self.num_cuts = int(getattr(self.config, "num_cuts", 4))        # K
        self.lamda = float(getattr(self.config, "lamda", 0.0))
        self.learning_rate = float(getattr(self.config, "learning_rate", 1e-5))
        self.weight_decay = float(getattr(self.config, "weight_decay", 0.0))
        self.num_epochs = int(getattr(self.config, "num_epochs", getattr(self.config, "epochs", 100)))
        if self.emb_dim < 1 or self.num_trees < 1 or self.num_cuts < 1 or self.num_epochs < 1:
            raise ValueError("TAM emb_dim, num_trees, num_cuts, num_epochs must be >= 1")

    def _truncated_adjs(self, raw_adj_np, x_np, t):
        """Sequential K-step NSGT truncation for tree t of a single graph."""
        rng = np.random.default_rng(self._seed + t)
        adjs = []
        cur = raw_adj_np
        for _ in range(self.num_cuts):
            cur = nsgt_truncate(cur, x_np, rng)
            adjs.append(cur)
        return adjs

    def train(self, samples):
        log_section("Training Phase - TAM")
        samples = [(np.asarray(x, dtype=np.float32), np.asarray(a, dtype=np.float32)) for x, a in samples]
        if not samples:
            raise ValueError("TAM received no training samples.")
        input_dim = samples[0][0].shape[-1]
        if any(x.ndim != 2 or x.shape[-1] != input_dim for x, _ in samples):
            raise ValueError("TAM training samples must have shape [agents, embedding_dim]")

        # raw_adj = A + I (used for affinity + regularization + as truncation seed).
        raw_adjs = [a + np.eye(a.shape[0], dtype=np.float32) for _, a in samples]

        # Precompute truncated adjacency matrices: trunc[t][k][sample_idx] -> (n, n)
        trunc = [[None] * self.num_cuts for _ in range(self.num_trees)]
        for t in range(self.num_trees):
            rng = np.random.default_rng(self._seed + t)
            cur = list(raw_adjs)
            for k in range(self.num_cuts):
                cur = [nsgt_truncate(a, x, rng) for a, (x, _) in zip(cur, samples)]
                trunc[t][k] = cur

        self.models = []
        xs = [torch.from_numpy(x) for x, _ in samples]
        raws = [torch.from_numpy(a) for a in raw_adjs]

        total_nets = self.num_trees * self.num_cuts
        for t in range(self.num_trees):
            for k in range(self.num_cuts):
                net = LAMNet(input_dim, self.emb_dim).to(self.device)
                optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                adjs = trunc[t][k]
                for epoch in range(self.num_epochs):
                    net.train()
                    total = 0.0
                    for idx in range(len(samples)):
                        x = xs[idx].to(self.device)
                        raw_adj = raws[idx].to(self.device)
                        adj_norm = _sym_normalize(torch.from_numpy(adjs[idx]).to(self.device))
                        h, feat1, _ = net(x, adj_norm)
                        loss = -local_affinity(h, raw_adj).sum() + self.lamda * reg_loss(feat1, raw_adj)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total += loss.item()
                    avg = total / max(1, len(samples))
                    if epoch % max(1, self.num_epochs // 10) == 0 or epoch == self.num_epochs - 1:
                        log_info(f"  LAMNet [{t * self.num_cuts + k + 1}/{total_nets}] "
                                 f"epoch {epoch + 1}/{self.num_epochs} loss {avg:.6f}")
                net.eval()
                self.models.append((t, k, net))

        log_done(f"TAM training complete ({len(self.models)} LAMNets).")

    def predict(self, round_data, adj_matrix):
        """Score a single debate round; returns (flags, anomaly_scores)."""
        if not self.models:
            raise RuntimeError("TAM model is not initialized; train or load a checkpoint first.")

        if isinstance(round_data, np.ndarray):
            x_np = round_data
        else:
            x_np = np.asarray([a["st_embedding"] for a in round_data], dtype=np.float32)
        adj_np = np.asarray(adj_matrix, dtype=np.float32)
        if adj_np.shape != (x_np.shape[0], x_np.shape[0]):
            raise ValueError(f"TAM adjacency must have shape {(x_np.shape[0], x_np.shape[0])}, got {adj_np.shape}")

        raw_adj_np = adj_np + np.eye(x_np.shape[0], dtype=np.float32)

        # Truncated adjacency per tree (fresh NSGT on the new graph).
        tree_adjs = {t: self._truncated_adjs(raw_adj_np, x_np, t) for t in range(self.num_trees)}

        with self._predict_lock, torch.no_grad():
            x = torch.from_numpy(x_np).float().to(self.device)
            raw_adj = torch.from_numpy(raw_adj_np).float().to(self.device)
            affinities = []
            for (t, k, net) in self.models:
                net.eval()
                adj_norm = _sym_normalize(torch.from_numpy(tree_adjs[t][k]).float().to(self.device))
                h, _, _ = net(x, adj_norm)
                affinities.append(local_affinity(h, raw_adj).cpu().numpy())

        aff_mean = np.mean(np.stack(affinities, axis=0), axis=0)
        lo, hi = aff_mean.min(), aff_mean.max()
        aff_norm = (aff_mean - lo) / (hi - lo) if hi > lo else np.zeros_like(aff_mean)
        scores = (1.0 - aff_norm).astype(float)

        n_agents = x_np.shape[0]
        threshold = getattr(self.config, "threshold", None)
        flags = np.zeros(n_agents, dtype=int)
        if threshold is not None:
            flags[scores > float(threshold)] = 1
        else:
            top_k = int(getattr(self.config, "top_k", 1))
            flags[np.argsort(-scores)[:min(top_k, n_agents)]] = 1
        return flags, scores

    def save_model(self, path):
        if not self.models:
            raise RuntimeError("Cannot save an uninitialized TAM model")
        checkpoint = {
            "num_trees": self.num_trees,
            "num_cuts": self.num_cuts,
            "input_dim": self.models[0][2].input_dim,
            "emb_dim": self.emb_dim,
            "state_dicts": [(t, k, net.state_dict()) for t, k, net in self.models],
        }
        torch.save(checkpoint, path)
        log_done(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.num_trees = checkpoint["num_trees"]
        self.num_cuts = checkpoint["num_cuts"]
        self.emb_dim = checkpoint["emb_dim"]
        self.models = []
        for t, k, sd in checkpoint["state_dicts"]:
            net = LAMNet(checkpoint["input_dim"], checkpoint["emb_dim"]).to(self.device)
            net.load_state_dict(sd)
            net.eval()
            self.models.append((t, k, net))
        log_info(f"Model loaded from {path}")


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

        samples = build_tam_samples(train_data)
        log_info(f"TAM data prepared: {len(samples)} round samples "
                 f"(feature dim {samples[0][0].shape[1]}).")

        log_info("Starting TAM training...")
        loop = TAMLoop(self.args)
        loop.train(samples)
        log_done("TAM model trained.")
        return {}, loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TAM: Truncated Affinity Maximization for Graph Anomaly Detection"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parsed = parser.parse_args()
    if not parsed.config:
        raise ValueError("--config <path_to_yaml> is required.")
    master = Master(parsed.config)
    _, model = master._run()
