from dataclasses import dataclass
import pickle
import sys
import numpy as np
import pickle
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Utils import load_config, load_config_from_path

@dataclass
class DataGenerationParams:
    anomaly_rate : float = 0.2
    anomaly_scale : float = 0.5
    
class TrainDataProcessor(DataGenerationParams):
    def __init__(self, params, target_topologies=None, rng_seed=None):
        super().__init__(**params)
        self.target_topologies = target_topologies
        self.rng = np.random.default_rng(rng_seed)

    def _extract_adj_matrix(self, record, fallback=None):
        """Extract adjacency matrix from common key variants."""
        if not isinstance(record, dict):
            return fallback
        adj = record.get('adj_matrix')
        if adj is None:
            adj = record.get('topology')
        if adj is None:
            adj = record.get('adjacency_matrix')
        if adj is None:
            return fallback
        return np.array(adj)

    def _build_single_debate_entry(self, debate, topology_name, adj_matrix):
        """Normalize one debate into the internal topology-like structure."""
        if not debate or 'debate_rounds' not in debate or adj_matrix is None:
            return None

        debate_emb = np.array([
            np.array([agent['st_embedding'] for agent in debate_round])
            for debate_round in debate['debate_rounds']
        ])

        n_agents = debate_emb.shape[1] if debate_emb.shape[0] > 0 else 0
        labels = np.zeros(n_agents, dtype=int)
        mal_indices = debate.get('malicious_agent_indexes')
        if mal_indices:
            labels[mal_indices] = 1

        return {
            'topology_name': topology_name,
            'adj_matrix': adj_matrix,
            'debates': [debate_emb],
            'labels': np.array([labels]),
        }
        
    def load_pkl(self, pkl_path):
        """Load pickle data and normalize to per-debate entries with their own adjacency."""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        processed_data = []
        for idx, entry in enumerate(data):
            # Legacy format: one entry groups many debates under a shared topology.
            if isinstance(entry, dict) and 'results' in entry:
                base_name = entry.get('topology_name', f"topology_{idx}")
                base_adj = self._extract_adj_matrix(entry)

                for debate_idx, debate in enumerate(entry['results']):
                    topology_name = debate.get('topology_name', base_name) if isinstance(debate, dict) else base_name
                    if self.target_topologies and topology_name not in self.target_topologies:
                        continue

                    debate_adj = self._extract_adj_matrix(debate, fallback=base_adj)
                    normalized = self._build_single_debate_entry(debate, topology_name, debate_adj)
                    if normalized is None:
                        continue

                    normalized['debate_id'] = f"{topology_name}_{debate_idx}"
                    processed_data.append(normalized)
                continue

            # New format: one entry is already one debate with its own topology.
            if not isinstance(entry, dict):
                continue
            topology_name = entry.get('topology_name', f"debate_{idx}")
            if self.target_topologies and topology_name not in self.target_topologies:
                continue

            debate_adj = self._extract_adj_matrix(entry)
            normalized = self._build_single_debate_entry(entry, topology_name, debate_adj)
            if normalized is None:
                continue
            normalized['debate_id'] = entry.get('debate_id', f"{topology_name}_{idx}")
            processed_data.append(normalized)

        self.data = processed_data
        
        # COmpute and save normalized adjacency matrices
        for topology in self.data:
            topology['norm_adj'] = self.normalize_adjacency(topology['adj_matrix'])
        
    def _get_data(self):
        return self.data
    
    def anomalize_sample(self, sample):
        x_noisy = sample + self.anomaly_scale * self.rng.standard_normal(sample.shape)
        return x_noisy
    
    def anomalize_debate(self, debate, anomaly_indices):
        for j, is_anomaly in enumerate(anomaly_indices):
            if is_anomaly == 1:
                # Add noise to all rounds for this agent
                debate[:, j, :] = debate[:, j, :] + self.anomaly_scale * self.rng.standard_normal(debate[:, j, :].shape)
        return debate

    def anomalize_data(self):
        for topology in self.data:
            n_debates = len(topology['debates'])
            n_agents = topology['debates'][0].shape[1] if n_debates > 0 else 0

            # anomaly_matrix: shape (n_debates, n_agents)
            anomaly_matrix = self.rng.binomial(1, self.anomaly_rate, size=(n_debates, n_agents))
            
            # Store anomaly labels for tracking (overwriting existing labels)
            topology['labels'] = anomaly_matrix

            for i in range(n_debates):
                topology['debates'][i] = self.anomalize_debate(topology['debates'][i], anomaly_matrix[i])
                
    def normalize_adjacency(self, adj_matrix):
        """Normalize adjacency matrix using symmetric normalization.
        Isolated nodes (degree 0) are given a zero row/column instead of
        raising a singular matrix error."""
        d = np.sum(adj_matrix, axis=1)
        d_inv_sqrt = np.zeros_like(d, dtype=float)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        return norm_adj
    
    def aggreagate_embeddings_round(self, debate_round, A_norm, adj_matrix):
        """Aggregate embeddings for a single debate round using the normalized adjacency matrix following BlindGuard indications."""
        size = debate_round.shape[0]
        h_graph = 1 / size * np.sum(debate_round, axis=0) # debate_round=(num_agents, embedding_dim)
        r = []
        for i in range(size):
            neighbor_indices = np.where(adj_matrix[i] == 1)[0]
            if len(neighbor_indices) == 0:
                h_neigh = np.zeros(debate_round.shape[1])
            else:
                h_neigh = sum(A_norm[i, j] * debate_round[j, :] for j in neighbor_indices)
            combined_feature = np.hstack((debate_round[i, :], h_neigh, h_graph))
            r.append(combined_feature)
        return np.array(r)
    
    def generate_aggregated_embeddings(self):
        """Generate aggregated embeddings for all debates in all topologies."""
        for topology in self.data:
            A_norm = topology['norm_adj']
            adj_matrix = topology['adj_matrix']
            aggregated_debates = []
            for debate in topology['debates']:
                aggregated_rounds = []
                for debate_round in debate:
                    aggregated_round = self.aggreagate_embeddings_round(debate_round, A_norm, adj_matrix)
                    aggregated_rounds.append(aggregated_round)
                aggregated_debates.append(np.array(aggregated_rounds))
            topology['aggregated_debates'] = aggregated_debates
            
    def _get_flattened_data(self, balance: bool = True):
        # Returns the data in a format ready to feed into the NN
        r = []
        for topology in self.data:
            flattened_debates = []
            labels = []
            
            # Use aggregated debates if available, else raw debates
            source_debates = topology.get('aggregated_debates', topology['debates'])
            source_labels = topology.get('labels')
            
            if source_labels is None:
                continue
                
            for i, debate in enumerate(source_debates):
                agent_labels = source_labels[i]  # Changed from source_labels[i, :] to handle ragged arrays
                for round_index in range(debate.shape[0]):
                    for agent_index in range(debate.shape[1]):
                        flattened_debates.append(debate[round_index, agent_index, :])
                        labels.append(agent_labels[agent_index])
            flattened_debates = np.array(flattened_debates)
            labels = np.array(labels)
            if balance:
                # Balance the dataset
                normal_indices = np.where(labels == 0)[0]
                anomaly_indices = np.where(labels == 1)[0]
                n_normals = len(normal_indices)
                n_anomalies = len(anomaly_indices)
                if n_anomalies == 0 or n_normals == 0:
                    continue  # Skip if one class is missing

                # Downsample both classes to the minority size for robust balancing.
                n_per_class = min(n_normals, n_anomalies)
                chosen_normal_indices = self.rng.choice(normal_indices, size=n_per_class, replace=False)
                chosen_anomaly_indices = self.rng.choice(anomaly_indices, size=n_per_class, replace=False)
                selected_indices = np.concatenate((chosen_normal_indices, chosen_anomaly_indices), axis=0)
                selected_indices = self.rng.permutation(selected_indices)
                flattened_debates = flattened_debates[selected_indices]
                labels = labels[selected_indices]
            r.append(
                {
                    'topology_name': topology['topology_name'],
                    'embeddings': flattened_debates,
                    'labels': labels
                }
            )
            
        return r

    def _get_flattened_data_rounds(self):
        # Returns data grouped by rounds to preserve context
        # Output structure: List of topologies
        # Each topology dict:
        #   'topology_name': str
        #   'rounds': List of dicts, where each dict is a round:
        #       'embeddings': (n_agents, embedding_dim)
        #       'labels': (n_agents,)
        
        r = []
        for topology in self.data:
            topology_rounds = []
            
            # Use aggregated debates if available, else raw debates
            source_debates = topology.get('aggregated_debates', topology['debates'])
            source_labels = topology.get('labels')
            
            if source_labels is None:
                continue

            for i, debate in enumerate(source_debates):
                # debate shape: (n_rounds, n_agents, embedding_dim)
                agent_labels = source_labels[i]  # Changed from source_labels[i, :] to handle ragged arrays
                
                for round_index in range(debate.shape[0]):
                    round_embeddings = debate[round_index, :, :] # shape: (n_agents, embedding_dim)
                    
                    # Labels are constant per agent across rounds in a debate
                    round_labels = agent_labels
                    
                    topology_rounds.append({
                        'embeddings': round_embeddings,
                        'labels': round_labels
                    })
            
            r.append({
                'topology_name': topology['topology_name'],
                'rounds': topology_rounds
            })
        return r
            
class SCLDataset(Dataset):
    """PyTorch dataset for SCL training."""
    
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings  # np.array of shape (N, embedding_dim)
        self.labels = labels  # np.array of shape (N,)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        features = self.embeddings[idx]
        label = self.labels[idx]
        return torch.from_numpy(features).float(), torch.tensor(label).long()
    
class TwoLayerMLP(nn.Module):
    """Two-layer MLP for embedding generation."""
    
    def __init__(self, input_dim, hidden_dim, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, dim=1)
        return x
    
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    # def forward(self, embeddings, labels):
    #     """
    #     embeddings: (N, D) normalized
    #     labels: (N,)
    #     """
    #     device = embeddings.device
    #     N = embeddings.shape[0]

    #     sim = torch.matmul(embeddings, embeddings.T) / self.temperature
    #     logits_mask = torch.ones_like(sim) - torch.eye(N, device=device)
    #     sim = sim * logits_mask

    #     labels = labels.view(-1, 1)
    #     positives = torch.eq(labels, labels.T).float().to(device)
    #     positives = positives * logits_mask

    #     exp_sim = torch.exp(sim) * logits_mask
    #     log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    #     mean_log_prob_pos = (positives * log_prob).sum(dim=1) / (positives.sum(dim=1) + 1e-9)
    #     loss = -mean_log_prob_pos.mean()
    #     return loss
    
    def forward(self, embeddings, labels):
        """
        embeddings: (N, D) 
        labels: (N,)
        """
        device = embeddings.device
        N = embeddings.shape[0]

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Cosine similarity matrix
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask self-similarity
        logits_mask = torch.ones_like(sim) - torch.eye(N, device=device)
        sim = sim * logits_mask

        # Positive mask
        labels = labels.view(-1, 1)
        positives = torch.eq(labels, labels.T).float().to(device)
        positives = positives * logits_mask

        # Log-softmax denominator
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # Mean over positives
        mean_log_prob_pos = (positives * log_prob).sum(dim=1) / (positives.sum(dim=1) + 1e-9)

        loss = -mean_log_prob_pos.mean()
        return loss


            
class SCLTopologyLoop:
    """Unified SCL model - train and inference (for test)."""
    
    def __init__(self, args):
        import threading
        self.args = args
        self.model = None
        self._predict_lock = threading.Lock()
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, topology_data: Dict[str, Any]):
        """Train model and evaluate on test set.
        
        Args:
            topology_data: Single topology dict from TrainDataProcessor._get_flattened_data
                          with keys: 'topology_name', 'embeddings', 'labels'
        """
        
        print("="*60)
        print(f"TRAINING PHASE - Topology: {topology_data['topology_name']}")
        print("="*60)
        
        X_full = topology_data['embeddings']
        y_full = topology_data['labels']
        
        # Validation split
        split_seed = getattr(self.args, 'split_seed', self.args.seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=self.args.val_split, random_state=split_seed, stratify=y_full
        )
        
        # Create dataset and dataloader
        dataset_train = SCLDataset(X_train, y_train)
        dataloader_seed = getattr(self.args, 'dataloader_seed', self.args.seed)
        train_generator = torch.Generator()
        train_generator.manual_seed(dataloader_seed)
        dataloader_train = DataLoader(dataset_train, batch_size=self.args.batch_size, shuffle=True, generator=train_generator)
        
        dataset_val = SCLDataset(X_val, y_val)
        dataloader_val = DataLoader(dataset_val, batch_size=self.args.batch_size, shuffle=False)
        
        # Initialize model
        self.model = TwoLayerMLP(input_dim=self.args.input_dim, hidden_dim=self.args.hidden_dim, emb_dim=self.args.emb_dim)
        self.model.to(self.device)
        
        criterion = SupConLoss(temperature=self.args.temperature)
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate, 
            weight_decay=self.args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.scheduler_t_max, eta_min=1e-5)
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        
        # Training loop
        num_epochs = self.args.num_epochs
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in dataloader_train:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)

                optimizer.zero_grad()
                embeddings = self.model(x)
                loss = criterion(embeddings, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation Step
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in dataloader_val:
                    x = batch[0].float().to(self.device)
                    y = batch[1].long().to(self.device)
                    embeddings = self.model(x)
                    loss = criterion(embeddings, y)
                    total_val_loss += loss.item()
            
            scheduler.step()
            avg_loss = total_loss / len(dataloader_train)
            avg_val_loss = total_val_loss / len(dataloader_val) if len(dataloader_val) > 0 else 0.0
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} [BEST]")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Load best model (kept in memory, no temporary file written).
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            print(f"\nTraining complete. Best model restored with validation loss: {best_val_loss:.6f}")
        else:
            print("\nTraining complete. No validation improvement snapshot was captured.")
                        
    def _reset_model(self):
        self.model = None
        
    # Now we are going to define methods that enable later live inference on LLM debate
    def predict(self, debate_round, adj_matrix, top_k=1):
        """Predict anomalies for a round using aggregated graph embeddings.
    
        Args:
            debate_round: List of dicts with 'st_embedding' key
            adj_matrix: Adjacency matrix for the topology (n_agents x n_agents)
            top_k: Number of agents to flag
    
        Returns:
            predictions: (n_agents,) array of 0/1 labels
            anomaly_scores: (n_agents,) array of anomaly scores
        """
        # Ensure adj_matrix is a numpy array (JSON loads it as a plain list)
        adj_matrix = np.array(adj_matrix)
        
        self.model.eval()
        
        # Step 1: Aggregate embeddings using adjacency matrix (same as training prep)
        raw_embeddings = np.array([agent['st_embedding'] for agent in debate_round])
        
        # Normalize adjacency (safe for isolated nodes with degree 0)
        d = np.sum(adj_matrix, axis=1)
        d_inv_sqrt = np.zeros_like(d, dtype=float)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        
        # Aggregate
        aggregated = []
        n_agents = raw_embeddings.shape[0]
        h_graph = (1/n_agents) * np.sum(raw_embeddings, axis=0)
        
        for i in range(n_agents):
            neighbor_indices = np.where(adj_matrix[i] == 1)[0]
            if len(neighbor_indices) == 0:
                h_neigh = np.zeros(raw_embeddings.shape[1])
            else:
                h_neigh = sum(A_norm[i, j] * raw_embeddings[j, :] for j in neighbor_indices)
            combined = np.hstack((raw_embeddings[i, :], h_neigh, h_graph))
            aggregated.append(combined)
        
        aggregated = np.array(aggregated)
        
        # Step 2: Pass through model — serialized via lock to avoid cuBLAS contention
        with self._predict_lock:
            self.model.cpu().eval()
            with torch.no_grad():
                x = torch.from_numpy(aggregated).float()
                embeddings = self.model(x).numpy()
        
        # Step 3: Compute anomaly scores
        similarity_matrix = cosine_similarity(embeddings)
        anomaly_scores = np.array([
            -(1/n_agents)*sum([similarity_matrix[i,j] for j in range(n_agents) if j != i]) 
            for i in range(n_agents)])
        
        flags = np.argsort(-anomaly_scores)[:top_k]
        predictions = np.zeros(n_agents, dtype=int)
        predictions[flags] = 1
        
        # Higher score => more anomalous/malicious (aligned with predictions and AUROC).
        return predictions, anomaly_scores

    def save_model(self, path):
        """Save the trained model and architecture to disk."""
        if self.model is not None:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.args.input_dim,
                'hidden_dim': self.args.hidden_dim,
                'emb_dim': self.args.emb_dim,
            }
            torch.save(checkpoint, path)
            print(f"Model saved to {path}")
        else:
            print("No model to save.")

    def load_model(self, path):
        """Load a trained model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = TwoLayerMLP(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            emb_dim=checkpoint['emb_dim']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {path}")
        
    @classmethod
    def from_pretrained(cls, model_path, device='cuda'):
        """Load a pretrained model for inference without needing full training args."""
        import types
        
        args = types.SimpleNamespace(
            device=device        
            )
        
        instance = cls(args)
        instance.load_model(model_path)
        return instance
    
class Master:
    def __init__(self, config_path):
        self.args = load_config_from_path(config_path)
        
    def _run(self):
        if self.args.save_model and not self.args.save_path:
            raise ValueError("If --save_model is True, you must provide a --save_path to save the model.")
        
        # Set random seeds for reproducibility
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
        
        # Update DataGenerationParams with self.args
        data_params = vars(DataGenerationParams())
        data_params['anomaly_rate'] = self.args.anomaly_rate
        data_params['anomaly_scale'] = self.args.anomaly_scale
        data_seed = getattr(self.args, 'data_seed', self.args.seed)

        print("Loading and processing training data...")
        train_data = TrainDataProcessor(data_params, target_topologies=self.args.topologies, rng_seed=data_seed)
        train_data.load_pkl(self.args.pkl_train)
        
        if self.args.anomalize_data:
            train_data.anomalize_data()
            
        train_data.generate_aggregated_embeddings()
        train_data = train_data._get_flattened_data(balance=not self.args.no_balance)
        
        print("Train data processed. Starting training...")
        scl_loop = SCLTopologyLoop(self.args)
        print("Training combined model across all topologies...")
        combined_topology = {
            'topology_name': 'combined',
            'embeddings': np.concatenate(tuple(t['embeddings'] for t in train_data), axis=0),
            'labels': np.concatenate(tuple(t['labels'] for t in train_data), axis=0)
        }
        print("Class distribution on combined training data:", np.bincount(combined_topology['labels']))
        scl_loop.train(combined_topology)
        print('Combined model trained.')
        if self.args.save_model:
            scl_loop.save_model(self.args.save_path)
            
        # Keep same return shape expected by MainEvaluation.
        return {}, scl_loop

if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description="Supervised Contrastive Learning for BlindGuard Anomaly Detection")
    arguments.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')
    parsed_config = arguments.parse_args()
    args = load_config(parsed_config)
    
    if args.save_model and not args.save_path:
        raise ValueError("If --save_model is True, you must provide a --save_path to save the model.")
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Update DataGenerationParams with args
    data_params = vars(DataGenerationParams())
    data_params['anomaly_rate'] = args.anomaly_rate
    data_params['anomaly_scale'] = args.anomaly_scale
    data_seed = getattr(args, 'data_seed', args.seed)

    print("Loading and processing training data...")
    train_data = TrainDataProcessor(data_params, target_topologies=args.topologies, rng_seed=data_seed)
    train_data.load_pkl(args.pkl_train)
    
    if args.anomalize_data:
        train_data.anomalize_data()
        
    train_data.generate_aggregated_embeddings()
    train_data = train_data._get_flattened_data(balance=not args.no_balance)
    
    print("Train data processed. Starting training...")
    scl_loop = SCLTopologyLoop(args)
    
    # So far only save combined model, since the other makes less sense (has bad performance)
    print("Training combined model across all topologies...")
    combined_topology = {
        'topology_name': 'combined',
        'embeddings': np.concatenate(tuple(t['embeddings'] for t in train_data), axis=0),
        'labels': np.concatenate(tuple(t['labels'] for t in train_data), axis=0)
    }
    print("Class distribution on combined training data:", np.bincount(combined_topology['labels']))
    scl_loop.train(combined_topology)
    print('Combined model trained.')
    if args.save_model:
        scl_loop.save_model(args.save_path)



