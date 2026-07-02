import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader
import umap
from sklearn.cluster import KMeans
from Utils import load_config_from_path

def _pad_agent_windows(per_agent_lists, device=None):
    flat_agents = []
    for agent in per_agent_lists:
        flat = []
        for t in agent:
            t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
            if t.dim() == 2:
                for i in range(t.shape[0]):
                    flat.append(t[i])
            else:
                flat.append(t)
        flat_agents.append(flat)

    M_max = max(len(e) for e in flat_agents if len(e) > 0)
    d_model = flat_agents[0][0].shape[-1] if flat_agents[0] else 0

    padded_agents = []
    for agent in flat_agents:
        if len(agent) == 0:
            padded = torch.zeros((M_max, d_model), dtype=torch.float32)
        else:
            stacked = torch.stack(agent, dim=0)
            M_i = stacked.shape[0]
            if M_i < M_max:
                mean_emb = stacked.mean(dim=0, keepdim=True)
                repeats = M_max - M_i
                padding = mean_emb.repeat(repeats, 1)
                padded = torch.cat([stacked, padding], dim=0)
            else:
                padded = stacked
        padded_agents.append(padded)

    out = torch.stack(padded_agents, dim=0)
    if device is not None:
        out = out.to(device)
    return out


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d

def load_config_from_path(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_ns(config_dict)

class TrainDataLoader:
    def __init__(self, target_topologies, pkl_path):
        self.target_topologies = target_topologies
        self.data = self.load_pkl(pkl_path)
        
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
    
    def load_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
        
        processed_data = []
        # We first process the data to get a list format        
        for i, record in enumerate(data): 
            if self.target_topologies and record['topology_name'] not in self.target_topologies:
                continue
            
            results = record.get('results', {})
            for debate in results:
                topology = debate['topology']
                embeddings = []
                agent_ids = []
                round_ids = []
                
                for round_id, round_data in enumerate(debate['debate_rounds']):
                    for agent_id, agent_data in enumerate(round_data):
                        agent_embeddings = agent_data['window_embeddings']
                        # This should be a list or array of embeddings
                        for i in range(len(agent_embeddings)):
                            embeddings.append(agent_embeddings[i])
                            agent_ids.append(agent_id)
                            round_ids.append(round_id)
                            
                processed_data.append({
                    'topology': topology,
                    'embeddings': embeddings,
                    'agent_ids': agent_ids,
                    'round_ids': round_ids
                })
                
        # In theory this has the same format as the output .pkl of my prepare_data.py        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        graph_data = self.data[idx]
        topology = torch.tensor(graph_data['topology'], dtype=torch.float32)
        
        num_agents = topology.shape[0]
        embeddings = graph_data['embeddings']
        agent_ids = graph_data['agent_ids']
        
        agent_embeddings = [[] for _ in range(num_agents)]
        
        for emb, agent_id in zip(embeddings, agent_ids):
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            agent_embeddings[agent_id].append(emb)
            
        # Return list of variable-length tensors (should be same length bbut...)
        
        return {
            "embeddings": agent_embeddings, # List of lists of tensors, one per agent
            "adjacency_matrix": topology,
            "graph_id": idx
        }
        
    def collate_fn(self, batch):
        r = []
        for graph in batch:
            embeddings = graph['embeddings']
            adj = graph['adjacency_matrix']
            node_tensor = _pad_agent_windows(embeddings)
            r.append({
                "embeddings" : node_tensor,
                "adjacency_matrix" : adj,
                "graph_id" : graph['graph_id']
            })
        return r
    
class IntraNodeSelfAttention(torch.nn.Module):
    """This module of the model implements self-attention
    between windows of the same agent."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        self.norm = torch.nn.LayerNorm(d_model)
        
    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(out + x)
    
class InterNodeGraphAttention(torch.nn.Module):
    """ This module implements graph attention between windows
    of different agents, using the adjacency matrix to mask attention."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = num_heads,
            batch_first = True
        )
        self.norm = torch.nn.LayerNorm(d_model)
        
    def forward(self, x, adj):
        num_nodes, _, d = x.shape
        out = torch.zeros_like(x)
        device = x.device
        
        for i in range(num_nodes):
            neighbors = adj[i].nonzero(as_tuple=False).squeeze(-1)
            if len(neighbors.shape) == 0:
                neighbors = neighbors.unsqueeze(0)
            neighbors = torch.cat([neighbors, torch.tensor([i], device=device)])
            
            neighbors = torch.unique(neighbors)
            context = x[neighbors] # (N, M, d)
            context = context.reshape(-1, d) # (N*M, d)
            
            query = x[i].unsqueeze(0) # (1, M, d)
            
            attn_out, _ = self.attn(
                query,
                context.unsqueeze(0),           # (1, N*M, d)
                context.unsqueeze(0)            # (1, N*M, d)
            )
            out[i] = self.norm(attn_out.squeeze(0) + x[i])   # back to (M, d)
            
        return out
    
class MultiScaleLoss(torch.nn.Module):
    """This implements the contrastive loss with a granular approach:
    - node-level: embeddings of same node must me similar
    - graph-level: embeddings of same graph must be similar
    - global-level: all bening embeddings must be similar"""
    
    def __init__(self, temperature = 0.07,
                 node_weight = 1.0,
                 graph_weight = 0.5,
                 global_weight = 0.1):
        super().__init__()
        self.temperature = temperature
        self.node_weight = node_weight
        self.graph_weight = graph_weight
        self.global_weight = global_weight
        
    def forward(self, embeddings_list, graph_indices):
        """
        Args:
            embeddings_list: List of tensors, each of shape (num_nodes, M, embedding_dim)
            graph_indices: List indicating which graph each embedding batch belongs to
        
        Returns:
            Scalar loss combining node-level, graph-level, and global losses
        """
        embeddings = []
        node_ids = []
        graph_ids = []
        
        node_counter = 0
        for graph_id, emb_tensor in zip(graph_indices, embeddings_list):
            num_nodes, M, d = emb_tensor.shape # M is the number of windows per node
            emb_flat = emb_tensor.reshape(num_nodes * M, d)
            embeddings.append(emb_flat)
            
            node_ids_graph = np.repeat(np.arange(num_nodes), M) + node_counter
            node_ids.extend(node_ids_graph)
            graph_ids.extend([graph_id] * (num_nodes * M))
            node_counter += num_nodes

        embeddings = torch.cat(embeddings, dim=0) # (total_nodes*M, d)
        device = embeddings.device
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) # to put in norm 1
        
        node_ids = torch.from_numpy(np.array(node_ids, dtype=np.int64)).to(device)
        graph_ids = torch.from_numpy(np.array(graph_ids, dtype=np.int64)).to(device)

        # COmpute similarity matrix for later loss computation
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, -10.0, 10.0)
        
        # Node-level loss
        node_mask = (node_ids.unsqueeze(0) == node_ids.unsqueeze(1)) & (graph_ids.unsqueeze(0) == graph_ids.unsqueeze(1))
        node_mask.fill_diagonal_(False)
        
        log_probs = similarity_matrix - torch.logsumexp(similarity_matrix, dim=1, keepdim=True)
        node_loss = -(
            log_probs[node_mask.bool()].mean()
        )
        
        graph_mask = (graph_ids[:, None] == graph_ids[None, :]) & ~node_mask
        graph_mask.fill_diagonal_(False)
        
        graph_loss = -(log_probs[graph_mask].mean())
            
        global_center = embeddings.mean(dim=0, keepdim=True)
        global_center = torch.nn.functional.normalize(global_center, dim=1)

        global_sim = embeddings @ global_center.T / self.temperature
        
        global_logits = global_sim.squeeze()

        global_log_probs = global_logits - torch.logsumexp(global_logits, dim=0)

        global_loss = -global_log_probs.mean()
        
        total_loss = (
            self.node_weight * node_loss +
            self.graph_weight * graph_loss +
            self.global_weight * global_loss
        )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return total_loss
        
class GraphEmbedder(torch.nn.Module):
    """Complete model combining all the previous modules."""
    def __init__(self, 
                 d_model, 
                 intraNode_layers,
                 intraNode_heads, 
                 interNode_layers, 
                 interNode_heads):
        super().__init__()
        
        self.node_attention_layers = torch.nn.ModuleList([
            IntraNodeSelfAttention(d_model, intraNode_heads) for _ in range(intraNode_layers)
        ])
        self.graph_attention_layers = torch.nn.ModuleList([
            InterNodeGraphAttention(d_model, interNode_heads) for _ in range(interNode_layers)
        ])
        
    def forward(self, X, adj):
        for layer in self.node_attention_layers:
            X = layer(X)
        for layer in self.graph_attention_layers:
            X = layer(X, adj)
        
        # Now normalize embeddings to hypershpere
        num_nodes, M, d = X.shape
        X = X.reshape(num_nodes * M, d)
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        X = X.reshape(num_nodes, M, d)
        
        return X
    
class KMeansCluster:
    def __init__(self, device, config, n_benign_clusters=1):
        self.random_state = config.random_state
        self.device = device
        self.threshold = config.threshold
        self.n_benign_clusters = n_benign_clusters
        self.config = config
        
    def get_classes(self, embeddings):
        # Implementation for getting labels
        max_k = min(self.config.max_k, max(2, len(embeddings)//2))
        inertias = []
        classes =  []
        for k in range(1, max_k+1):  
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
            classes.append(kmeans.labels_)
            
        if len(inertias) < 2:
            return np.array(classes[0] if classes else None)
        
        else:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            elbow_k = np.argmax(second_diffs) + 2
            elbow_id = min(max(2, elbow_k), max_k)
            return np.array(classes[elbow_id - 1])

    def classify_embeddings(self, embeddings):
        """Deals with complete classification and flagging.
        Must also apply the uMAP before the clustering.
        Must intake only ONE round information.
        Could be an idea to prestablish the uMAP transform with train data for faster inferernce."""
        
        n_agents, n_windows, dim_emb = embeddings.shape

        # 1. Flatten to (n_agents * n_windows, dim_emb) for UMAP + clustering
        flat_emb = embeddings.cpu().numpy().reshape(-1, dim_emb)

        # 2. Dimensionality reduction
        reducer = umap.UMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            random_state=self.random_state,
            metric=self.config.umap_metric,
            n_jobs = 1
        )
        reduced_emb = reducer.fit_transform(flat_emb)

        # 3. Cluster
        clusters = self.get_classes(reduced_emb)
        if clusters is None:
            return np.zeros(n_agents, dtype=bool)

        # 4. Identify benign vs suspicious clusters
        cluster_counts = np.bincount(clusters)
        benign_clusters = set(np.argsort(cluster_counts)[-self.n_benign_clusters:])
        suspicious_clusters = set(range(len(cluster_counts))) - benign_clusters

        # 5. Per-agent flagging
        is_suspicious_emb = np.isin(clusters, list(suspicious_clusters))  # (n_agents * n_windows,)
        is_suspicious_emb = is_suspicious_emb.reshape(n_agents, n_windows) # (n_agents, n_windows)

        suspicious_ratio = is_suspicious_emb.mean(axis=1) 
        # (n_agents,)
        return (suspicious_ratio >= self.threshold).astype(int), suspicious_ratio, reduced_emb

class WindowBreakerModel:
    def __init__(self, config):
        self.config = config
        self.model = GraphEmbedder(
            d_model = config.d_model,
            intraNode_layers = config.intraNode_layers,
            intraNode_heads = config.intraNode_heads,
            interNode_layers = config.interNode_layers,
            interNode_heads = config.interNode_heads
        )
        self.device = self.config.device if hasattr(self.config, 'device') else 'cpu'
        
    def train_step(self, loss_fn, optimizer, batch):
        embeddings = []
        graph_ids = []
        for graph_id, sample in enumerate(batch):
            node_emb = sample['embeddings'].to(self.device)
            adj = sample['adjacency_matrix'].to(self.device)
            
            node_points_cloud = self.model(node_emb, adj)
            embeddings.append(node_points_cloud)
            graph_ids.append(graph_id)
            
        loss = loss_fn(embeddings, graph_ids)
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def val_step(self, loss_fn, batch):
        embeddings = []
        graph_ids = []
        for graph_id, sample in enumerate(batch):
            node_emb = sample['embeddings'].to(self.device)
            adj = sample['adjacency_matrix'].to(self.device)
            node_points_cloud = self.model(node_emb, adj)
            embeddings.append(node_points_cloud)
            graph_ids.append(graph_id)
        loss = loss_fn(embeddings, graph_ids)
        return loss.item()
        
    def _train(self, train_data_loader):
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.config.learning_rate,
            weight_decay = self.config.weight_decay
        )
        self.scheduler = self.create_scheduler(self.optimizer, self.config.scheduler)
        loss_fn = MultiScaleLoss(
            temperature = self.config.temperature,
            node_weight = self.config.node_weight,
            graph_weight = self.config.graph_weight,
            global_weight = self.config.global_weight
        )

        # --- Validation split ---
        val_split = getattr(self.config, 'validation_split', 0.0)
        val_seed = getattr(self.config, 'validation_seed', 42)

        from torch.utils.data import Subset
        num_graphs = len(train_data_loader)
        num_val = max(1, int(num_graphs * val_split)) if val_split > 0 else 0
        num_train = num_graphs - num_val

        has_val = num_val > 0 and num_train > 0

        if has_val:
            rng = np.random.default_rng(val_seed)
            indices = np.arange(num_graphs)
            rng.shuffle(indices)
            train_indices = indices[:num_train].tolist()
            val_indices = indices[num_train:].tolist()

            train_dataset = Subset(train_data_loader, train_indices)
            val_dataset = Subset(train_data_loader, val_indices)

            print(f"[Data] Split {num_graphs} graphs: {num_train} train / {num_val} val (seed={val_seed})")
        else:
            train_dataset = train_data_loader
            val_dataset = None
            print(f"[Data] Using all {num_graphs} graphs for training (no validation split)")

        train_loader = DataLoader(
            train_dataset,
            batch_size = self.config.batch_size,
            shuffle = True,
            collate_fn = train_data_loader.collate_fn
        )

        if has_val:
            val_loader = DataLoader(
                val_dataset,
                batch_size = self.config.batch_size,
                shuffle = False,
                collate_fn = train_data_loader.collate_fn
            )

        best_val_loss = float('inf')
        patience = 0
        max_patience = self.config.scheduler.patience
        no_improve = 0

        for epoch in range(self.config.epochs):
            # --- Train ---
            self.model.train()
            train_epoch_loss = 0.0
            num_train_batches = 0
            for batch in train_loader:
                loss = self.train_step(loss_fn, self.optimizer, batch)
                if not np.isnan(loss) and not np.isinf(loss) and loss>0:
                    train_epoch_loss += loss
                    num_train_batches += 1
            avg_train_loss = train_epoch_loss / num_train_batches if num_train_batches > 0 else 0.0

            # --- Validate ---
            if has_val:
                self.model.eval()
                val_epoch_loss = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        loss = self.val_step(loss_fn, batch)
                        if not np.isnan(loss) and not np.isinf(loss) and loss>0:
                            val_epoch_loss += loss
                            num_val_batches += 1
                avg_val_loss = val_epoch_loss / num_val_batches if num_val_batches > 0 else float('inf')
            else:
                avg_val_loss = avg_train_loss

            self.scheduler.step(avg_val_loss)

            lr = self.optimizer.param_groups[0]['lr']

            is_best = False
            if best_val_loss == float('inf'):
                improvement = True
            else:
                improvement = (best_val_loss - avg_val_loss) / best_val_loss > self.config.improve_threshold

            if improvement:
                best_val_loss = avg_val_loss
                patience = 0
                no_improve = 0
                torch.save(self.model.state_dict(), self.config.save_path)
                is_best = True
                print(
                    f"[Epoch {epoch+1}/{self.config.epochs}] "
                    f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | "
                    f"best_val={best_val_loss:.6f} | "
                    f"patience={patience}/{max_patience} | lr={lr:.6e} | [BEST]"
                )
            else:
                patience += 1
                no_improve += 1
                print(
                    f"[Epoch {epoch+1}/{self.config.epochs}] "
                    f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | "
                    f"best_val={best_val_loss:.6f} | "
                    f"patience={patience}/{max_patience} | lr={lr:.6e}"
                )

            if patience >= max_patience and not is_best:
                lr *= self.config.scheduler.factor
                for gr in self.optimizer.param_groups:
                    gr['lr'] = lr
                patience = 0

            if no_improve >= self.config.early_stop:
                print("[Done] Early stopping triggered")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        self.model.load_state_dict(torch.load(self.config.save_path, map_location=self.device))
        self.model.eval()

        del self.scheduler
        del self.optimizer
        del train_loader
        if has_val:
            del val_loader
        
    def predict(self, graph_data, adj_matrix):
        per_agent = [
            [torch.as_tensor(v) for v in node['window_embeddings']]
            for node in graph_data
        ]
        embeddings = _pad_agent_windows(per_agent, device=self.device)
        adj = torch.tensor(adj_matrix).to(self.device)
        classifier = KMeansCluster(self.device, self.config.kmeans_config)
        with torch.no_grad():
            node_points_cloud = self.model(embeddings, adj)
            flags, anomaly_score, embeddings = classifier.classify_embeddings(node_points_cloud.cpu())
        return flags, anomaly_score, embeddings
        
    def create_scheduler(self, optimizer, scheduler_config):
        factor = scheduler_config.factor
        patience = scheduler_config.patience
        threshold = scheduler_config.threshold
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, 
            threshold=threshold
        )

class Master:
    def __init__(self, config_path):
        self.config = load_config_from_path(config_path)
        
    def _run(self):
        """Must init the config and then run all the train process
        Must return an object with the predict attribute to make predictions"""
        train_loader = TrainDataLoader(
            target_topologies=self.config.data.target_topologies,
            pkl_path=self.config.data.train_pkl_path
        )
        model = WindowBreakerModel(self.config)
        model._train(train_loader)
        return {}, model