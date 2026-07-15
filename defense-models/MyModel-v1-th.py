import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import DataLoader
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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

        return {
            "embeddings": agent_embeddings,
            "adjacency_matrix": topology,
            "graph_id": idx
        }

    def collate_fn(self, batch):
        r = []
        for graph in batch:
            embeddings = graph['embeddings']
            adj = graph['adjacency_matrix']
            window_counts = [len(agent) for agent in embeddings]
            node_tensor = _pad_agent_windows(embeddings)
            r.append({
                "embeddings": node_tensor,
                "adjacency_matrix": adj,
                "graph_id": graph['graph_id'],
                "window_counts": window_counts
            })
        return r


class IntraNodeSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(out + x)


class InterNodeGraphAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
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
            context = x[neighbors]
            context = context.reshape(-1, d)

            query = x[i].unsqueeze(0)

            attn_out, _ = self.attn(
                query,
                context.unsqueeze(0),
                context.unsqueeze(0)
            )
            out[i] = self.norm(attn_out.squeeze(0) + x[i])

        return out


class GraphRegularizedHypersphereLoss(torch.nn.Module):
    def __init__(self,
                 cohesion_weight=1.0,
                 neighbor_weight=0.5,
                 graph_separation_weight=0.1,
                 temperature=0.5,
                 huber_delta=1.0):
        super().__init__()
        self.cohesion_weight = cohesion_weight
        self.neighbor_weight = neighbor_weight
        self.graph_separation_weight = graph_separation_weight
        self.temperature = temperature
        self.huber_delta = huber_delta

    def _huber(self, x, delta):
        abs_x = x.abs()
        return torch.where(abs_x <= delta,
                           0.5 * x ** 2,
                           delta * (abs_x - 0.5 * delta))

    def forward(self, embeddings_list, graph_indices, adj_matrices):
        device = embeddings_list[0].device
        total_windows = 0
        cohesion_numer = 0.0
        neighbor_numer = 0.0
        neighbor_count = 0
        graph_centroids = []

        for emb_tensor, adj in zip(embeddings_list, adj_matrices):
            num_nodes, M, d = emb_tensor.shape
            adj = adj.to(device)

            agent_center = emb_tensor.mean(dim=1, keepdim=True)
            agent_center = torch.nan_to_num(agent_center, nan=0.0)
            cohesion_numer += (emb_tensor - agent_center).pow(2).sum()
            total_windows += num_nodes * M

            agent_center = agent_center.squeeze(1)

            for i in range(num_nodes):
                nbrs = (adj[i] > 0).nonzero(as_tuple=False).squeeze(-1)
                if nbrs.dim() == 0:
                    nbrs = nbrs.unsqueeze(0)
                nbrs = nbrs[nbrs != i]
                if len(nbrs) == 0:
                    continue
                nbr_center = agent_center[nbrs].mean(dim=0)
                dist = torch.norm(agent_center[i] - nbr_center)
                neighbor_numer += self._huber(dist, self.huber_delta)
                neighbor_count += 1

            graph_centroids.append(agent_center.mean(dim=0, keepdim=True))

        L_cohesion = cohesion_numer / total_windows if total_windows > 0 else torch.tensor(0.0, device=device)
        L_neighbor = neighbor_numer / neighbor_count if neighbor_count > 0 else torch.tensor(0.0, device=device)

        graph_centroids = torch.cat(graph_centroids, dim=0)
        graph_centroids = torch.nn.functional.normalize(graph_centroids, p=2, dim=1)
        graph_centroids = torch.nan_to_num(graph_centroids, nan=0.0)
        ng = graph_centroids.shape[0]

        if ng > 1:
            sim_g = torch.mm(graph_centroids, graph_centroids.t())
            sim_g = sim_g.clamp(-1.0, 1.0)
            pairwise_sq_g = (2 - 2 * sim_g).clamp(min=1e-8, max=4.0)
            mask_g = ~torch.eye(ng, dtype=torch.bool, device=device)
            pairwise_sq_g = pairwise_sq_g[mask_g]
            sep_val = torch.exp(-pairwise_sq_g / self.temperature)
            L_separation = torch.log(sep_val.mean() + 1e-8)
        else:
            L_separation = torch.tensor(0.0, device=device)

        total_loss = (self.cohesion_weight * L_cohesion +
                      self.neighbor_weight * L_neighbor +
                      self.graph_separation_weight * L_separation)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(1e-8, device=device, requires_grad=True)

        return total_loss


class GraphEmbedder(torch.nn.Module):
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
        max_k = min(self.config.max_k, max(2, len(embeddings)//2))
        inertias = []
        classes = []
        for k in range(1, max_k + 1):
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
        n_agents, n_windows, dim_emb = embeddings.shape

        flat_emb = embeddings.cpu().numpy().reshape(-1, dim_emb)

        scaler = StandardScaler()
        standardized = scaler.fit_transform(flat_emb)

        reducer = umap.UMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            random_state=self.random_state,
            metric=self.config.umap_metric,
            n_jobs=1,
        )
        reduced_emb = reducer.fit_transform(standardized)

        clusters = self.get_classes(reduced_emb)

        if clusters is None:
            return np.zeros(n_agents, dtype=int), np.zeros(n_agents), reduced_emb, None

        cluster_counts = np.bincount(clusters)
        largest_size = cluster_counts.max()

        benign_clusters = {
            i for i, count in enumerate(cluster_counts)
            if count == largest_size
        }
        suspicious_clusters = set(range(len(cluster_counts))) - benign_clusters

        benign_mask = np.isin(clusters, list(benign_clusters))

        if benign_mask.any():
            benign_centroid = reduced_emb[benign_mask].mean(axis=0)
        else:
            benign_centroid = np.zeros(reduced_emb.shape[1])

        is_suspicious = np.isin(clusters, list(suspicious_clusters))
        distances = np.zeros(len(clusters))

        if is_suspicious.any():
            raw_distances = np.linalg.norm(
                reduced_emb[is_suspicious] - benign_centroid,
                axis=1,
            )

            d_min = raw_distances.min()
            d_max = raw_distances.max()

            if d_max > d_min:
                distances[is_suspicious] = (
                    (raw_distances - d_min) /
                    (d_max - d_min)
                )
            else:
                distances[is_suspicious] = 0.0

        agent_scores = distances.reshape(n_agents, n_windows).mean(axis=1)

        predictions = (agent_scores >= self.threshold).astype(int)

        return predictions, agent_scores, reduced_emb, clusters


class WindowBreakerModel:
    def __init__(self, config):
        self.config = config
        self.model = GraphEmbedder(
            d_model=config.d_model,
            intraNode_layers=config.intraNode_layers,
            intraNode_heads=config.intraNode_heads,
            interNode_layers=config.interNode_layers,
            interNode_heads=config.interNode_heads
        )
        self.device = self.config.device if hasattr(self.config, 'device') else 'cpu'

    def train_step(self, loss_fn, optimizer, batch):
        embeddings = []
        graph_ids = []
        adj_matrices = []
        for graph_id, sample in enumerate(batch):
            node_emb = sample['embeddings'].to(self.device)
            adj = sample['adjacency_matrix'].to(self.device)
            node_points_cloud = self.model(node_emb, adj)
            embeddings.append(node_points_cloud)
            graph_ids.append(graph_id)
            adj_matrices.append(adj)
        loss = loss_fn(embeddings, graph_ids, adj_matrices)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def val_step(self, loss_fn, batch):
        embeddings = []
        graph_ids = []
        adj_matrices = []
        for graph_id, sample in enumerate(batch):
            node_emb = sample['embeddings'].to(self.device)
            adj = sample['adjacency_matrix'].to(self.device)
            node_points_cloud = self.model(node_emb, adj)
            embeddings.append(node_points_cloud)
            graph_ids.append(graph_id)
            adj_matrices.append(adj)
        loss = loss_fn(embeddings, graph_ids, adj_matrices)
        return loss.item()

    def _run_threshold_validation(self, train_data_loader, thresh_val_indices, classifier):
        print()
        print("=" * 72)
        print("[Threshold Validation] Running post-training threshold analysis")
        print("=" * 72)
        print("  All graphs in this set are benign — any anomaly is a false positive.")
        print()

        self.model.eval()
        per_graph_thresholds = []

        with torch.no_grad():
            for idx in thresh_val_indices:
                graph_data = train_data_loader.data[idx]
                topology = graph_data['topology']
                embeddings = graph_data['embeddings']
                agent_ids = graph_data['agent_ids']
                round_ids = graph_data['round_ids']

                num_agents = len(topology)
                unique_rounds = sorted(set(round_ids))

                print(f"  [Graph {idx}] {len(unique_rounds)} round(s)")

                for round_id in unique_rounds:
                    round_mask = [i for i, r in enumerate(round_ids) if r == round_id]
                    round_embeddings_list = [embeddings[i] for i in round_mask]
                    round_agent_ids_list = [agent_ids[i] for i in round_mask]

                    agent_embeddings = [[] for _ in range(num_agents)]
                    for emb, aid in zip(round_embeddings_list, round_agent_ids_list):
                        if not isinstance(emb, torch.Tensor):
                            emb = torch.tensor(emb, dtype=torch.float32)
                        agent_embeddings[aid].append(emb)

                    window_counts = [len(a) for a in agent_embeddings]

                    padded = _pad_agent_windows(agent_embeddings, device=self.device)
                    adj = torch.tensor(topology, dtype=torch.float32).to(self.device)

                    node_points_cloud = self.model(padded, adj)
                    _, anomaly_scores, _, clusters = classifier.classify_embeddings(
                        node_points_cloud.cpu()
                    )

                    graph_max = float(anomaly_scores.max())
                    per_graph_thresholds.append(graph_max)

                    cluster_counts = np.bincount(clusters)
                    num_clusters = len(cluster_counts)

                    print(f"    [Round {round_id}] max anomaly score = {graph_max:.6f}")
                    print(f"      windows per agent: {window_counts}")
                    print(f"      agent anomaly scores: {np.array2string(anomaly_scores, precision=6, suppress_small=True)}")
                    print(f"      clusters: {num_clusters} total, sizes: {cluster_counts.tolist()}")

                print()

            if per_graph_thresholds:
                arr = np.array(per_graph_thresholds)
                n = len(arr)
                median = np.median(arr)
                mad = np.median(np.abs(arr - median))
                k = 1.5
                optimal_threshold = median + k * mad
                print(f"  Summary over {n} graph-round max scores:")
                print(f"    median = {median:.6f}, MAD = {mad:.6f}")
                print(f"    using threshold = median + {k}*MAD = {optimal_threshold:.6f} for live evaluation")
                print()
            else:
                optimal_threshold = None

        return per_graph_thresholds, optimal_threshold

    def _train(self, train_data_loader):

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = self.create_scheduler(self.optimizer, self.config.scheduler)
        loss_fn = GraphRegularizedHypersphereLoss(
            cohesion_weight=getattr(self.config, 'cohesion_weight', 1.0),
            neighbor_weight=getattr(self.config, 'neighbor_weight', 0.5),
            graph_separation_weight=getattr(self.config, 'graph_separation_weight', 0.1),
            temperature=getattr(self.config, 'loss_temperature', 0.5),
            huber_delta=getattr(self.config, 'huber_delta', 1.0)
        )

        val_split = getattr(self.config, 'validation_split', 0.0)
        val_seed = getattr(self.config, 'validation_seed', 42)
        thresh_val_split = getattr(self.config, 'threshold_validation_split', 0.0)

        from torch.utils.data import Subset
        num_graphs = len(train_data_loader)

        num_thresh_val = max(1, int(num_graphs * thresh_val_split)) if thresh_val_split > 0 else 0
        remaining = num_graphs - num_thresh_val
        num_val = max(1, int(remaining * val_split)) if val_split > 0 and remaining > 0 else 0
        num_train = remaining - num_val

        has_thresh_val = num_thresh_val > 0 and num_train > 0
        has_val = num_val > 0 and num_train > 0

        if has_thresh_val or has_val:
            rng = np.random.default_rng(val_seed)
            indices = np.arange(num_graphs)
            rng.shuffle(indices)
            train_indices = indices[:num_train].tolist() if num_train > 0 else []
            val_indices = indices[num_train:num_train + num_val].tolist() if num_val > 0 else []
            thresh_val_indices = indices[num_train + num_val:].tolist() if num_thresh_val > 0 else []

            train_dataset = Subset(train_data_loader, train_indices) if train_indices else None
            val_dataset = Subset(train_data_loader, val_indices) if val_indices else None
            thresh_val_dataset = Subset(train_data_loader, thresh_val_indices) if thresh_val_indices else None

            parts = []
            if num_train > 0:
                parts.append(f"{num_train} train")
            if num_val > 0:
                parts.append(f"{num_val} val")
            if num_thresh_val > 0:
                parts.append(f"{num_thresh_val} thresh_val")
            print(f"[Data] Split {num_graphs} graphs: {' / '.join(parts)} (seed={val_seed})")
        else:
            train_dataset = train_data_loader
            val_dataset = None
            thresh_val_dataset = None
            print(f"[Data] Using all {num_graphs} graphs for training (no splits)")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=train_data_loader.collate_fn
        )

        if has_val:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=train_data_loader.collate_fn
            )

        if has_thresh_val:
            thresh_val_loader = DataLoader(
                thresh_val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=train_data_loader.collate_fn
            )

        best_val_loss = float('inf')
        patience = 0
        max_patience = self.config.scheduler.patience
        no_improve = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            train_epoch_loss = 0.0
            num_train_batches = 0
            for batch in train_loader:
                loss = self.train_step(loss_fn, self.optimizer, batch)
                if not np.isnan(loss) and not np.isinf(loss):
                    train_epoch_loss += loss
                    num_train_batches += 1
            avg_train_loss = train_epoch_loss / num_train_batches if num_train_batches > 0 else 0.0

            if has_val:
                self.model.eval()
                val_epoch_loss = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        loss = self.val_step(loss_fn, batch)
                        if not np.isnan(loss) and not np.isinf(loss):
                            val_epoch_loss += loss
                            num_val_batches += 1
                if num_val_batches > 0:
                    avg_val_loss = val_epoch_loss / num_val_batches
                else:
                    avg_val_loss = avg_train_loss
                    print(f"  [Warn] All validation batches had invalid loss, falling back to train loss")
            else:
                avg_val_loss = avg_train_loss

            self.scheduler.step(avg_val_loss)

            lr = self.optimizer.param_groups[0]['lr']

            is_best = False
            if best_val_loss == float('inf'):
                improvement = True
            else:
                improvement = (best_val_loss - avg_val_loss) / abs(best_val_loss) > self.config.improve_threshold

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

        if has_thresh_val:
            thresh_classifier = KMeansCluster(self.device, self.config.kmeans_config)
            per_graph_thresholds, optimal_threshold = self._run_threshold_validation(
                train_data_loader, thresh_val_indices, thresh_classifier
            )
            self.computed_threshold = optimal_threshold
            print(f"[Threshold] Computed inference threshold from training data: {self.computed_threshold:.6f}")
        else:
            per_graph_thresholds = []
            self.computed_threshold = None

        del self.scheduler
        del self.optimizer
        del train_loader
        if has_val:
            del val_loader
        if has_thresh_val:
            del thresh_val_loader

    def predict(self, graph_data, adj_matrix):
        per_agent = [
            [torch.as_tensor(v) for v in node['window_embeddings']]
            for node in graph_data
        ]
        raw_text = [
            node.get("raw_windows", None) for node in graph_data
        ]
        embeddings = _pad_agent_windows(per_agent, device=self.device)
        adj = torch.tensor(adj_matrix).to(self.device)
        classifier = KMeansCluster(self.device, self.config.kmeans_config)
        if getattr(self, 'computed_threshold', None) is not None:
            classifier.threshold = self.computed_threshold
        with torch.no_grad():
            node_points_cloud = self.model(embeddings, adj)
            flags, anomaly_score, embeddings, clusters = classifier.classify_embeddings(node_points_cloud.cpu())
        # if raw_text[0] == None:
        #     return flags, anomaly_score, embeddings, clusters
        # else:
        #     return flags, anomaly_score, embeddings, clusters, raw_text
            return flags, anomaly_score

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
        train_loader = TrainDataLoader(
            target_topologies=self.config.data.target_topologies,
            pkl_path=self.config.data.train_pkl_path
        )
        model = WindowBreakerModel(self.config)
        model._train(train_loader)
        result_meta = {}
        if model.computed_threshold is not None:
            model.config.kmeans_config.threshold = model.computed_threshold
            result_meta["computed_threshold"] = model.computed_threshold
        return result_meta, model
