"""PREM-v2: PREM variant matching the reference implementation found in the
BlindGuard repo (PI/Prem_gad.py).

Compared to PREM.py this variant:

* Aggregates neighbors with **mean** semantics (``scatter_mean``) instead of the
  symmetric-normalized **sum** used by the canonical paper. In matrix form one
  propagation step is ``D^-1/2 (D^-1 A) D^-1/2`` where ``D = degree(A)`` (no
  explicit self-loop is added).
* Initializes the discriminator linear layers with ``xavier_uniform_`` and zero
  bias (as done by the reference ``_init_weights``).
* Keeps the same anonymized ego removal (zero the propagation diagonal) and the
  same contrastive training objective (Eq. 5-8), but with the negative pairs
  actually used -- the reference's ``train_un2.py`` computes the negatives and
  then discards them, which makes its loss degenerate.

Data intake (sentence embeddings via BlindGuard's loader) and the framework
``Master`` / ``predict`` interface are identical to PREM.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys

_PREM_DIR = Path(__file__).resolve().parent
sys.path.append(str(_PREM_DIR.parent))
sys.path.insert(0, str(_PREM_DIR))

from LoggingUtils import log_done, log_info
from Utils import load_config_from_path

from PREM import (
    PREMDiscriminator,
    PREMTopologyLoop,
    build_prem_dataset,
)
from BlindGuard import TrainDataProcessor


def mean_anonymized_propagation(adj_matrix, k):
    """Anonymized propagation with mean aggregation (reference Prem_gad.py).

    One step mirrors ``scatter_mean``: ``X <- D^-1/2 * mean_neighbors(X)`` with a
    ``D^-1/2`` scaling applied both before and after the mean, i.e.
    ``S = D^-1/2 (D^-1 A) D^-1/2`` where ``D = degree(A)`` (no self-loop). The
    k-step matrix has its diagonal zeroed so neighbor features exclude the ego.
    """
    adj = np.asarray(adj_matrix, dtype=np.float64)
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    deg_inv = np.zeros_like(degree)
    deg_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    deg_inv[nonzero] = 1.0 / degree[nonzero]
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    d_inv = np.diag(deg_inv)
    d_inv_sqrt = np.diag(deg_inv_sqrt)
    s = d_inv_sqrt @ d_inv @ adj @ d_inv_sqrt
    s_k = np.linalg.matrix_power(s, int(k))
    np.fill_diagonal(s_k, 0.0)
    return s_k


class PREM2Discriminator(PREMDiscriminator):
    """Ego-neighbor matching network with the reference's xavier init."""

    def __init__(self, input_dim, emb_dim):
        super().__init__(input_dim, emb_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class PREMTopologyLoopV2(PREMTopologyLoop):
    """PREM-v2 loop: same training/inference, mean-based propagation."""

    def __init__(self, args):
        super().__init__(
            args,
            propagation_fn=mean_anonymized_propagation,
            discriminator_cls=PREM2Discriminator,
        )


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
        prem_data = build_prem_dataset(
            train_data, prop_steps, propagation_fn=mean_anonymized_propagation
        )
        log_info(f"PREM-v2 data prepared: {prem_data['ego'].shape[0]} ego/neighbor node pairs "
                 f"(feature dim {prem_data['ego'].shape[1]}).")

        log_info("Starting PREM-v2 training...")
        prem_loop = PREMTopologyLoopV2(self.args)
        prem_loop.train(prem_data)
        log_done("PREM-v2 model trained.")
        return {}, prem_loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PREM-v2: mean-aggregation PREM variant"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parsed = parser.parse_args()
    if not parsed.config:
        raise ValueError("--config <path_to_yaml> is required.")
    master = Master(parsed.config)
    _, model = master._run()
