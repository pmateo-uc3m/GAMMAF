import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch, Data
import pickle
import argparse
import random
import numpy as np
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from Utils import load_config, load_config_from_path

class DataProcessor:
    def __init__(self, target_topologies=None):
        self.target_topologies = target_topologies

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
        
    def load_pkl(self, pkl_path):
        """Should load the debates data in the proper format to feed into the model (geometric graph)."""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        def convert_round_to_geo_graph(round_messages, attacker_idxes, edge_idx_tensor):
            node_features_s = []
            x_i_prime = [] # Mean token embeddings per agent (can be tensor)
            x_i_j_prime = [] # Contextualized raw token embeddings per agent
            for msg in round_messages:
                if isinstance(msg, dict):
                    st_embedding = msg.get("st_embedding")
                    node_features_s.append(
                        st_embedding
                        )
                    token_embeddings_list = msg.get("tk_embedding")
                    contextualized_tokens = []
                    for token_emb in token_embeddings_list:
                        contextualized_tokens.append(
                            np.array(token_emb) + np.array(msg.get("st_embedding"))
                        )
                    x_i_prime.append(np.array(contextualized_tokens).mean(axis=0)) # Mean pool to get fixed-size representation
                    x_i_j_prime.append(contextualized_tokens)
                        
            # COnextualized tokens is like x'_i_j

            node_features_s = np.array(node_features_s)
            x_i_prime = np.array(x_i_prime)
            
            x_s = torch.FloatTensor(node_features_s)
            
            # WE cant save t features as tensor because each agent has different number of tokens
            x_t = torch.FloatTensor(x_i_prime)
            
            # Better approach is to save the mean poolos x_token_prime as tensor and raw contextualized token embeddings as list of numpy arrays in the graph data, so we can use them later for covariance detection
            
            
            num_nodes = x_s.shape[0]
            y = torch.zeros(num_nodes, dtype=torch.long)
            for idx in attacker_idxes:
                if 0 <= idx < num_nodes:
                    y[idx] = 1
                    
            graph_s = Data(
                x=x_s,
                edge_index=edge_idx_tensor,
                y=y,
                num_nodes=num_nodes
            )
            
            graph_t = Data(
                x=x_t,
                edge_index=edge_idx_tensor,
                y=y,
                num_nodes=num_nodes,
                per_token=x_i_j_prime
            )
            
            return graph_s, graph_t
        
        def convert_to_geo_graph(debate_data, adj_matrix : np.ndarray):
            """Convert all the rounds of one exection to PyTorch Geometric graph format.
            We do this so that then the data is feadable to a GNN."""
            edge_index = [[i, j] for i in range(adj_matrix.shape[0]) for j in range(adj_matrix.shape[1]) if adj_matrix[i, j] != 0]
            if len(edge_index) > 0 and len(edge_index[0]) > 0:
                edge_index_tensor = torch.LongTensor(edge_index).t().contiguous()  # Transpose to [2, num_edges]
            else:
                edge_index_tensor = torch.LongTensor([[], []])

            attacker_idxes = debate_data.get("malicious_agent_indexes", [])
            results = []
            
            for round in debate_data['debate_rounds']:
                messages = [{"st_embedding": msg.get("st_embedding"), "tk_embedding": msg.get("tk_embedding")} for msg in round]
                graph_s, graph_t = convert_round_to_geo_graph(messages, attacker_idxes, edge_index_tensor)
                results.append((graph_s, graph_t))
                
            return results
            
        processed_data = []
        for idx, entry in enumerate(data):
            # Legacy format: one record with many debates under 'results'.
            if isinstance(entry, dict) and 'results' in entry:
                base_name = entry.get('topology_name', f"topology_{idx}")
                base_adj = self._extract_adj_matrix(entry)

                for debate_idx, debate in enumerate(entry['results']):
                    topology_name = debate.get('topology_name', base_name) if isinstance(debate, dict) else base_name
                    if self.target_topologies and topology_name not in self.target_topologies:
                        continue

                    debate_adj = self._extract_adj_matrix(debate, fallback=base_adj)
                    if debate_adj is None or not isinstance(debate, dict) or 'debate_rounds' not in debate:
                        continue

                    o = {
                        'topology_name': topology_name,
                        'adj_matrix': debate_adj,
                        'debate_id': f"{topology_name}_{debate_idx}",
                        'results': [convert_to_geo_graph(debate, debate_adj)],
                    }
                    processed_data.append(o)
                continue

            # New format: one record is one debate with its own topology.
            if not isinstance(entry, dict):
                continue
            topology_name = entry.get('topology_name', f"debate_{idx}")
            if self.target_topologies and topology_name not in self.target_topologies:
                continue

            debate_adj = self._extract_adj_matrix(entry)
            if debate_adj is None or 'debate_rounds' not in entry:
                continue

            o = {
                'topology_name': topology_name,
                'adj_matrix': debate_adj,
                'debate_id': entry.get('debate_id', f"{topology_name}_{idx}"),
                'results': [convert_to_geo_graph(entry, debate_adj)],
            }
            processed_data.append(o)
            
        self.data = processed_data
        return self.data
    
# At this point the data is supposed to be ready for the models
class XGGuard(torch.nn.Module):
    def __init__(self, feat_dim_s, feat_dim_t, hidden_dim):
        super(XGGuard, self).__init__()
        self.gcn_s1 = GCNConv(feat_dim_s, hidden_dim)
        self.gcn_s2 = GCNConv(hidden_dim, feat_dim_s)
        
        self.gcn_t1 = GCNConv(feat_dim_t, hidden_dim)
        self.gcn_t2 = GCNConv(hidden_dim, feat_dim_t)
        
    def get_prototypes_sentence(self, x, batch_index):
        return global_mean_pool(x, batch_index)
    
    def get_prototypes_token(self, x, batch_index):
        mean_per_agent = torch.stack([emb.mean(dim=0) for emb in x])
        return global_mean_pool(mean_per_agent, batch_index)
    
    # def compute_scores_sentence(self, x, prototypes, batch_index):
    #     """Compute distance of sentence embedding to sentence theme prototype."""
    #     node_prototypes = prototypes[batch_index]
    #     x_norm = F.normalize(x, p=2, dim=1)  # MAybe i should remove this (not normalize)
    #     p_norm = F.normalize(node_prototypes, p=2, dim=1)
    #     cos_sim = torch.sum(x_norm * p_norm, dim=1, keepdim=True)
    #     return cos_sim
    
    def compute_scores_sentence(self, x, prototypes, batch_index):
        """Compute distance of sentence embedding to sentence theme prototype."""
        node_prototypes = prototypes[batch_index]
        cos_sim = torch.sum(x * node_prototypes, dim=1, keepdim=True)
        return cos_sim
    
    
    def compute_scores_token(self, token_emb_list, prototypes, batch_index):
        """Vectorized version - compute all scores at once."""
        device = prototypes.device
        
        # Flatten all tokens into one tensor with agent indices
        all_tokens = []
        agent_indices = []
        for i, agent_tokens in enumerate(token_emb_list):
            if agent_tokens.numel() > 0:
                all_tokens.append(agent_tokens)
                agent_indices.extend([i] * len(agent_tokens))
        
        if not all_tokens:
            return torch.full((len(token_emb_list), 1), 0.5, device=device)
        
        # Stack all tokens: [total_tokens, 384]
        all_tokens_tensor = torch.cat(all_tokens, dim=0)
        agent_indices = torch.tensor(agent_indices, device=device)
        
        # Get prototypes for each token's agent
        agent_ids = batch_index[agent_indices]  # [total_tokens]
        token_prototypes = prototypes[agent_ids]  # [total_tokens, 384]
        
        distances = torch.sum(all_tokens_tensor * token_prototypes, dim=1) # [total_tokens]
        
        # Group by agent using scatter_mean
        scores = torch.zeros(len(token_emb_list), device=device)
        scores.scatter_add_(0, agent_indices, distances)
        token_counts = torch.zeros(len(token_emb_list), device=device)
        token_counts.scatter_add_(0, agent_indices, torch.ones_like(agent_indices, dtype=torch.float))
        scores = scores / (token_counts + 1e-8)  # Mean per agent
        
        return scores.unsqueeze(1)  # [num_agents, 1]
    
    def fusion_logic(self, scores_s, scores_t, batch_index):
        """Per-graph normalization and fusion with sigmoid."""
        if batch_index is None:
            batch_index = torch.zeros(
                scores_s.shape[0],
                dtype=torch.long,
                device=scores_s.device)
            
        scores_hat_s_list = []
        scores_hat_t_list = []
        num_graphs = batch_index.max().item() + 1
        for graph_id in range(num_graphs):
            mask = (batch_index == graph_id)
            s_scores = scores_s[mask]
            t_scores = scores_t[mask]
            
            mean_s = s_scores.mean()
            std_s = s_scores.std() + 1e-10 # Deberia poner aqui unbiased?
            mean_t = t_scores.mean()
            std_t = t_scores.std() + 1e-10
        
            
            s_scores_hat = (s_scores - mean_s) / std_s
            t_scores_hat = (t_scores - mean_t) / std_t
            
            scores_hat_s_list.append(s_scores_hat)
            scores_hat_t_list.append(t_scores_hat)
            
        s_hat_s = torch.cat(scores_hat_s_list, dim=0)
        s_hat_t = torch.cat(scores_hat_t_list, dim=0)
        
        covariance = torch.mean(s_hat_s * s_hat_t)
        s_g = s_hat_s + (covariance * s_hat_t)
        
        s_prob = torch.sigmoid(s_g)
        return s_prob
    
    def fusion_logic_simple(self, scores_s, scores_t):
        """Global per-batch normalization and fusion with sigmoid."""
        # Normalize globally across entire batch (not per-graph)
        mean_s = scores_s.mean()
        std_s = scores_s.std() + 1e-10
        mean_t = scores_t.mean()
        std_t = scores_t.std() + 1e-10
        
        s_hat_s = (scores_s - mean_s) / std_s
        s_hat_t = (scores_t - mean_t) / std_t
        
        # print(f"[DEBUG FUSION] s_hat_s: min={s_hat_s.min():.4f}, max={s_hat_s.max():.4f}, mean={s_hat_s.mean():.4f}")
        # print(f"[DEBUG FUSION] s_hat_t: min={s_hat_t.min():.4f}, max={s_hat_t.max():.4f}, mean={s_hat_t.mean():.4f}")
        
        # Compute covariance between normalized scores
        covariance = torch.mean(s_hat_s * s_hat_t)
        s_g = s_hat_s + (covariance * s_hat_t)
        
        s_prob = torch.sigmoid(s_g)
        return s_prob
    
    def forward(self, batch_s, batch_t, debug=False):
        
        h_s = F.relu(self.gcn_s1(batch_s.x, batch_s.edge_index))
        h_s = self.gcn_s2(h_s, batch_s.edge_index)
        h_s = h_s + batch_s.x # Contains the encoding of all agents in batch
        
        # Now create token-level node features h_t for prototype computation using gcn
        h_t = F.relu(self.gcn_t1(batch_t.x, batch_t.edge_index))
        h_t = self.gcn_t2(h_t, batch_t.edge_index) #[T, 384]
        
        # Now get h_i_j^t by adding the per-token residual to the output of the GNN
        per_token_emb_list = []
        for agent_idx in range(batch_t.x.size(0)):
            graph_id = batch_s.batch[agent_idx].item()
            local_idx = agent_idx - batch_s.ptr[graph_id].item()
            raw_tokens = batch_t.per_token[graph_id][local_idx]
            
            x_prime_tokens = torch.from_numpy(np.array(raw_tokens)).float().to(batch_s.x.device)  # [T_i, 384]
            
            h_i_j_t = h_t[agent_idx:agent_idx+1] + x_prime_tokens  # [1,384] broadcast to [T_i, 384]
            per_token_emb_list.append(h_i_j_t)
        
        h_t = per_token_emb_list # List of [T_i, 384] tensors for each agent
        #Compute prototypes
        p_s = self.get_prototypes_sentence(h_s, batch_s.batch)
        p_t = self.get_prototypes_token(h_t, batch_t.batch)
        
        #Get sentence and token-level scores of the agent both shape [num_agents, 1]
        s_s_pos = self.compute_scores_sentence(h_s, p_s, batch_s.batch)
        s_t_pos = self.compute_scores_token(h_t, p_t, batch_s.batch)
        
        if debug:
            print(f"[DEBUG POSITIVE] s_s_pos: min={s_s_pos.min():.4f}, max={s_s_pos.max():.4f}, mean={s_s_pos.mean():.4f}")
            print(f"[DEBUG POSITIVE] s_t_pos: min={s_t_pos.min():.4f}, max={s_t_pos.max():.4f}, mean={s_t_pos.mean():.4f}")
        
        #Fuse both scores using global batch normalization
        s_pos = self.fusion_logic_simple(s_s_pos, s_t_pos)
        
        # Negative set creation: compare agent to random prototypes of other graphs in the batch
        
        # TODO: Include a seed for reproducibility
        num_graphs = batch_s.batch.max().item() + 1
        neg_graph_ids = []
        
        for node_graph_id in batch_s.batch:
            node_graph_id_val = node_graph_id.item()
            available_graphs = [g for g in range(num_graphs) if g != node_graph_id_val]
            if available_graphs:
                sampled_graph = available_graphs[
                    torch.randint(0, len(available_graphs), (1,)).item()
                    ]
            else:
                sampled_graph = node_graph_id_val
                
            neg_graph_ids.append(sampled_graph)
        
        neg_batch_index = torch.tensor(
            neg_graph_ids, 
            dtype=torch.long, 
            device=batch_s.batch.device)
        
        # Now we compute the negative scores
        s_s_neg = self.compute_scores_sentence(h_s, p_s, neg_batch_index)
        s_t_neg = self.compute_scores_token(h_t, p_t, neg_batch_index)
        
        if debug:
            print(f"[DEBUG NEGATIVE] s_s_neg: min={s_s_neg.min():.4f}, max={s_s_neg.max():.4f}, mean={s_s_neg.mean():.4f}")
            print(f"[DEBUG NEGATIVE] s_t_neg: min={s_t_neg.min():.4f}, max={s_t_neg.max():.4f}, mean={s_t_neg.mean():.4f}")
            print(f"[DEBUG] Sentence score margin (pos-neg): {(s_s_pos.mean() - s_s_neg.mean()):.4f}")
            print(f"[DEBUG] Token score margin (pos-neg): {(s_t_pos.mean() - s_t_neg.mean()):.4f}")
        
        s_neg = self.fusion_logic_simple(s_s_neg, s_t_neg)
        
        return s_pos, s_neg

class Loop:
    def __init__(self, model, config):
        import threading
        self.model = model
        self.config = config
        self._predict_lock = threading.Lock()
        
    def _train(self, train_loader, val_loader, device='cpu'):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0
            
            for batch_pair in train_loader:
                batch_s = batch_pair[0].to(device)
                batch_t = batch_pair[1].to(device)
                
                optimizer.zero_grad()
                
                # Debug: print scores for first batch of each epoch
                debug_batch = (epoch % 5 == 0)  # Print every 5 epochs
                s_pos, s_neg = self.model(batch_s, batch_t, debug=False) # set to debug_batch for debugging
                eps = 1e-8
                loss = -(torch.log(s_pos + eps) + self.config.alpha * torch.log(1 - s_neg + eps)).mean()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            
            if val_loader is not None:
                val_loss, val_metrics = self.validate_model(val_loader, self.config.alpha, device)
                val_msg = f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}"
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    print(f"{val_msg} [BEST]")
                else:
                    print(val_msg)
            else:
                print(f"Epoch {epoch+1}/{self.config.epochs} | Train Loss: {avg_loss:.6f}")

        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            print(f"\nTraining complete. Best model restored with validation loss: {best_val_loss:.6f}")
    
        return best_val_loss
    
    def validate_model(self, dataloader, alpha, device='cpu'):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        total_pos_score = 0
        total_neg_score = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_pair in dataloader:
                batch_s = batch_pair[0].to(device)
                batch_t = batch_pair[1].to(device)
                
                s_pos, s_neg = self.model(batch_s, batch_t, debug=False)
                
                eps = 1e-8
                loss = -(torch.log(s_pos + eps) + alpha * torch.log(1 - s_neg + eps)).mean()
                total_loss += loss.item()
                total_pos_score += s_pos.mean().item()
                total_neg_score += s_neg.mean().item()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches
        avg_pos_score = total_pos_score / num_batches
        avg_neg_score = total_neg_score / num_batches
        margin = avg_pos_score - avg_neg_score
        
        metrics = {
            'pos_score': avg_pos_score,
            'neg_score': avg_neg_score,
            'margin': margin
        }
        
        return avg_loss, metrics
    
    def _evaluate(self, evaluation_data, device='cpu', top_k=None):
        """Evaluate model on the evaluation data.
        
        Args:
            evaluation_data: Data to evaluate on
            device: Device to use
            top_k: If provided, label top-k agents with lowest s_pos WITHIN EACH GRAPH as 1, rest as 0
            
        Returns:
            Tuple of:
            - per_graph_predictions: List of prediction tensors, one per graph
            - per_graph_labels: List of label tensors, one per graph
            - per_graph_accuracy: List of accuracy values, one per graph
            - per_graph_raw_scores: List of raw s_pos scores (before top_k), one per graph
        """
        self.model.eval()
        per_graph_predictions = []
        per_graph_labels = []
        per_graph_accuracy = []
        per_graph_raw_scores = []
        
        with torch.no_grad():
            # evaluation_data comes from DataProcessor.load_pkl()
            for topology_data in evaluation_data:
                topology_name = topology_data['topology_name']
                
                for debate_rounds in topology_data['results']:  # Each debate
                    for graph_s, graph_t in debate_rounds:      # Each round (each graph)
                        # Preserve tokens_raw per graph before batching
                        tokens_raw_backup = graph_t.tokens_raw if hasattr(graph_t, 'tokens_raw') else []
                        # Wrap single graphs in a Batch so .batch attribute is populated
                        graph_s = Batch.from_data_list([graph_s]).to(device)
                        graph_t = Batch.from_data_list([graph_t]).to(device)
                        graph_t.tokens_raw_per_graph = [tokens_raw_backup]
                        
                        # Forward pass
                        s_pos, _ = self.model(graph_s, graph_t, debug=False)
                        
                        # Get labels
                        graph_labels = graph_s.y.cpu()
                        
                        # Store raw scores for AUROC computation
                        per_graph_raw_scores.append(s_pos.cpu().squeeze())
                        
                        # Apply top-k thresholding within this graph if specified
                        if top_k is not None:
                            # Get indices of top-k lowest scores within this graph
                            num_agents = s_pos.shape[0]
                            k = min(top_k, num_agents)
                            _, top_k_indices = torch.topk(s_pos.squeeze(), k=k, largest=False)
                            
                            # Create binary predictions for this graph: 1 for top-k lowest, 0 for rest
                            graph_predictions = torch.zeros_like(s_pos)
                            graph_predictions[top_k_indices] = 1
                            s_pos = graph_predictions
                        
                        # Calculate per-graph accuracy
                        accuracy = (s_pos.cpu().squeeze() == graph_labels).float().mean().item()
                        
                        # Store per-graph predictions and labels (stay grouped)
                        per_graph_predictions.append(s_pos.cpu().squeeze())
                        per_graph_labels.append(graph_labels)
                        per_graph_accuracy.append(accuracy)
                    
        return per_graph_predictions, per_graph_labels, per_graph_accuracy, per_graph_raw_scores
    
    def predict(self, round_data : dict, adj_matrix, top_k: int):
        """Predict the top-k anomalous agent from a single round (already embedded)
        
        round_data must have keys 'st_embedding' and 'tk_embedding'"""
        # Ensure adj_matrix is a numpy array (JSON loads it as a plain list)
        adj_matrix = np.array(adj_matrix)

        self.model.eval()
        n_agents = len(round_data)
        
        # Convert edge_index
        edge_index = [[i, j] for i in range(adj_matrix.shape[0]) 
                    for j in range(adj_matrix.shape[1]) if adj_matrix[i, j] != 0]
        if len(edge_index) > 0:
            edge_index_tensor = torch.LongTensor(edge_index).t().contiguous()
        else:
            edge_index_tensor = torch.LongTensor([[], []])
        
        # Extract embeddings
        st_embeddings = np.array([agent['st_embedding'] for agent in round_data])
        tk_embeddings = [agent['tk_embedding'] for agent in round_data]
        
        # Create x_t (mean-pooled token embeddings per agent)
        x_t_list = []
        for tk_emb in tk_embeddings:
            mean_tokens = np.array(tk_emb).mean(axis=0)  # Mean over tokens
            x_t_list.append(mean_tokens)
        x_t = np.array(x_t_list)
        
        # Create graph objects
        x_s = torch.FloatTensor(st_embeddings)
        x_t_tensor = torch.FloatTensor(x_t)
        
        graph_s = Data(x=x_s, edge_index=edge_index_tensor, y=torch.zeros(n_agents), num_nodes=n_agents)
        graph_t = Data(x=x_t_tensor, edge_index=edge_index_tensor, y=torch.zeros(n_agents), 
                    num_nodes=n_agents, per_token=tk_embeddings)
        
        # Wrap in batch (required for .batch attribute)
        batch_s = Batch.from_data_list([graph_s]).to('cpu')
        batch_t = Batch.from_data_list([graph_t]).to('cpu')
        batch_t.tokens_raw_per_graph = [tk_embeddings]
        
        # Inference — serialized via lock and model pinned to CPU to avoid cuBLAS contention
        with self._predict_lock:
            self.model.cpu().eval()
            with torch.no_grad():
                s_pos, _ = self.model(batch_s, batch_t, debug=False)
        
        # Get scores and top-k
        scores = s_pos.cpu().numpy().flatten()
        top_k_indices = np.argsort(scores)[:top_k]  # Smallest scores = most anomalous
        anomaly_scores = -scores  # Higher score => more anomalous/malicious.
        
        predictions = np.zeros(n_agents, dtype=int)
        predictions[top_k_indices] = 1
        
        return predictions, anomaly_scores
    
    def save_model(self, path):
        """Save the trained model and architecture to disk."""
        if self.model is not None:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'feat_dim_s': self.config.feat_dim_s if hasattr(self.config, 'feat_dim_s') else None,
                'feat_dim_t': self.config.feat_dim_t if hasattr(self.config, 'feat_dim_t') else None,
                'hidden_dim': self.config.hidden_dim if hasattr(self.config, 'hidden_dim') else None,
            }
            torch.save(checkpoint, path)
            print(f"Model saved to {path}")
        else:
            print("No model to save.")

    def load_model(self, path, device='cpu'):
        """Load a trained model from disk."""
        checkpoint = torch.load(path, map_location=device)
        
        self.model = XGGuard(
            feat_dim_s=checkpoint['feat_dim_s'],
            feat_dim_t=checkpoint['feat_dim_t'],
            hidden_dim=checkpoint['hidden_dim']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(cls, model_path, device='cpu'):
        """Load a pretrained model for inference without needing full training args."""
        import types
        
        # Load checkpoint to get architecture parameters
        checkpoint = torch.load(model_path, map_location=device)
        
        # Reconstruct config object with model architecture info
        config = types.SimpleNamespace(
            feat_dim_s=checkpoint['feat_dim_s'],
            feat_dim_t=checkpoint['feat_dim_t'],
            hidden_dim=checkpoint['hidden_dim']
        )
        
        # Create model instance
        model = XGGuard(
            feat_dim_s=checkpoint['feat_dim_s'],
            feat_dim_t=checkpoint['feat_dim_t'],
            hidden_dim=checkpoint['hidden_dim']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create and return Loop instance
        instance = cls(model, config)
        return instance
        
    
# Custom dataset for (graph_s, graph_t) pairs
class GeometricPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]


def create_geometric_dataloader(pairs, batch_size, shuffle=True, seed=None):
    """Create a dataloader for (graph_s, graph_t) pairs with proper batching."""
    dataset = GeometricPairDataset(pairs)
    
    def collate_fn(batch):
        batch_s_list = [pair[0] for pair in batch]
        batch_t_list = [pair[1] for pair in batch]
        batch_s = Batch.from_data_list(batch_s_list)
        batch_t = Batch.from_data_list(batch_t_list)
        
        tokens_raw_per_graph = []
        for graph_t in batch_t_list:
            if hasattr(graph_t, 'tokens_raw'):
                tokens_raw_per_graph.append(graph_t.tokens_raw)
            else:
                tokens_raw_per_graph.append([])
                
        batch_t.tokens_raw_per_graph = tokens_raw_per_graph
        return (batch_s, batch_t)
    
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, generator=generator)

class Master:
    """Class that manages all the train-evaluate-save logic.
    Should return the evaluation trace and the trained model (for possible live test)."""
    def __init__(self, config_path):
        self.args = load_config_from_path(config_path)
    
    def _run(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
        
        # Set device
        if self.args.device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.args.device
        print(f"Using device: {device}")
        
        print("Loading and processing training data...")
        train_processor = DataProcessor(target_topologies=self.args.topologies)
        train_data = train_processor.load_pkl(self.args.pkl_train)
        
        # Flatten train data into list of (graph_s, graph_t) pairs
        train_pairs = []
        for topology_data in train_data:
            for debate_rounds in topology_data['results']:
                for graph_s, graph_t in debate_rounds:
                    train_pairs.append((graph_s, graph_t))
        
        print(f"Total training samples: {len(train_pairs)}")
        
        # Split into train and validation with deterministic shuffling.
        split_seed = getattr(self.args, 'split_seed', self.args.seed)
        split_rng = np.random.default_rng(split_seed)
        indices = split_rng.permutation(len(train_pairs))
        split_idx = int(len(train_pairs) * (1 - self.args.val_split))
        train_set = [train_pairs[i] for i in indices[:split_idx]]
        val_set = [train_pairs[i] for i in indices[split_idx:]]
        
        # Create DataLoaders
        dataloader_seed = getattr(self.args, 'dataloader_seed', self.args.seed)
        train_loader = create_geometric_dataloader(train_set, self.args.batch_size, shuffle=True, seed=dataloader_seed)
        val_loader = create_geometric_dataloader(val_set, self.args.batch_size, shuffle=False) if len(val_set) > 0 else None
        
        print(f"Train samples: {len(train_set)}, Validation samples: {len(val_set)}")
        
        print("Training data processed. Starting training...")
        
        # Create model and training loop
        xg_guard_model = XGGuard(
            feat_dim_s=self.args.feat_dim_s,
            feat_dim_t=self.args.feat_dim_t,
            hidden_dim=self.args.hidden_dim
        ).to(device)
        
        @dataclass
        class TrainConfig:
            batch_size: int = self.args.batch_size
            learning_rate: int = self.args.learning_rate
            epochs: int = self.args.num_epochs
            alpha: float = self.args.alpha
            lr: float = self.args.learning_rate
            weight_decay: float = self.args.weight_decay
        
        config = TrainConfig()
        loop = Loop(xg_guard_model, config)
        
        print(f"Training XG-Guard model with config: batch_size={config.batch_size}, lr={config.lr}, epochs={config.epochs}")
        best_val_loss = loop._train(train_loader, val_loader, device=device)
        print('Model trained.')

        # Keep the same return shape expected by MainEvaluation.
        return {}, loop

if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description="XG-Guard Anomaly Detection with Graph Neural Networks")
    arguments.add_argument('--config', type=str, default=None, help='Path to YAML config file with all parameters')    
    parsed_config = arguments.parse_args()
    args = load_config(parsed_config)
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    print("Loading and processing training data...")
    train_processor = DataProcessor(target_topologies=args.topologies)
    train_data = train_processor.load_pkl(args.pkl_train)
    
    # Flatten train data into list of (graph_s, graph_t) pairs
    train_pairs = []
    for topology_data in train_data:
        for debate_rounds in topology_data['results']:
            for graph_s, graph_t in debate_rounds:
                train_pairs.append((graph_s, graph_t))
    
    print(f"Total training samples: {len(train_pairs)}")
    
    # Split into train and validation with deterministic shuffling.
    split_seed = getattr(args, 'split_seed', args.seed)
    split_rng = np.random.default_rng(split_seed)
    indices = split_rng.permutation(len(train_pairs))
    split_idx = int(len(train_pairs) * (1 - args.val_split))
    train_set = [train_pairs[i] for i in indices[:split_idx]]
    val_set = [train_pairs[i] for i in indices[split_idx:]]
    
    # Create DataLoaders
    dataloader_seed = getattr(args, 'dataloader_seed', args.seed)
    train_loader = create_geometric_dataloader(train_set, args.batch_size, shuffle=True, seed=dataloader_seed)
    val_loader = create_geometric_dataloader(val_set, args.batch_size, shuffle=False) if len(val_set) > 0 else None
    
    print(f"Train samples: {len(train_set)}, Validation samples: {len(val_set)}")
    
    print("Training data processed. Starting training...")
    
    # Create model and training loop
    xg_guard_model = XGGuard(
        feat_dim_s=args.feat_dim_s,
        feat_dim_t=args.feat_dim_t,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    @dataclass
    class TrainConfig:
        batch_size: int = args.batch_size
        learning_rate: int = args.learning_rate
        epochs: int = args.num_epochs
        alpha: float = args.alpha
        lr: float = args.learning_rate
        weight_decay: float = args.weight_decay
    
    config = TrainConfig()
    loop = Loop(xg_guard_model, config)
    
    print(f"Training XG-Guard model with config: batch_size={config.batch_size}, lr={config.lr}, epochs={config.epochs}")
    best_val_loss = loop._train(train_loader, val_loader, device=device)

    print('Model trained.')