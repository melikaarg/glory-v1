import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import numpy as np


class TrainDataset(IterableDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg):
        super().__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_input = news_input
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.gpu_num

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, np.array(mask, dtype='float32')

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        clicked_index, clicked_mask = self.pad_to_fix_len(self.trans_to_nindex(click_id), self.cfg.model.his_size)
        clicked_input = self.news_input[clicked_index]

        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        return clicked_input, clicked_mask, candidate_input, label

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)
    
    
class TrainGraphDataset(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors

    def line_mapper(self, line, sum_num_news):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # ------------------ Clicked News ----------------------
        # ------------------ News Subgraph ---------------------
        top_k = len(click_id)
        click_idx = self.trans_to_nindex(click_id)  
        source_idx = click_idx     
        for _ in range(self.cfg.model.k_hops) :
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)
        padded_maping_idx = F.pad(mapping_idx, (self.cfg.model.his_size-len(mapping_idx), 0), "constant", -1)

        
        # ------------------ Candidate News ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]  #[5, 5]
            candidate_neighbor_entity = np.zeros(((self.cfg.npratio+1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64) # [5*5, 20]
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio+1, self.cfg.model.entity_size *self.cfg.model.entity_neighbors) # [5, 5*20]
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
               sum_num_news+sub_news_graph.num_nodes

    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset: 
            subset = [0]
            
        subset = torch.tensor(subset, dtype=torch.long, device=device)
        
        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr, relabel_nodes=True, num_nodes=self.news_graph.num_nodes)
                    
        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k]+sum_num_nodes
    
    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    # if line.strip().split('\t')[3]:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news = self.line_mapper(line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))


                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                        clicked_graphs, mappings ,candidates, labels, candidate_entity_list, entity_mask_list  = [], [], [], [], [], []
                        sum_num_news = 0

                if (len(clicked_graphs) > 0):
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                    f.seek(0)


class TrainGraphDatasetWithFirstClustering(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, clusters):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors
        self.clusters = clusters

    def line_mapper(self, line, sum_num_news):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # ------------------ Clicked News ----------------------
        # Convert clicked news to indices
        click_idx = self.trans_to_nindex(click_id)

        # Get the cluster ID of the first clicked news article
        cluster_id = None
        for idx in click_idx:
            for cluster, nodes in self.clusters.items():
                if idx in nodes:
                    cluster_id = cluster
                    break
            if cluster_id is not None:
                break

        # If no cluster is found for the clicked news, handle as a special case (e.g., skip or assign default cluster)
        if cluster_id is None:
            # Handle cases where clicked news is not in any cluster
            pass

        # Get only neighbors within the same cluster
        top_k = len(click_id)
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                # Filter neighbors by cluster
                neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if n in self.clusters[cluster_id]]
                current_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # Build the subgraph using the filtered click_idx within the same cluster
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)
        padded_maping_idx = F.pad(mapping_idx, (self.cfg.model.his_size - len(mapping_idx), 0), "constant", -1)

        # ------------------ Candidate News ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]
            candidate_neighbor_entity = np.zeros(
                ((self.cfg.npratio + 1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio + 1,
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
            sum_num_news + sub_news_graph.num_nodes
    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset:
            subset = [0]

        subset = torch.tensor(subset, dtype=torch.long, device=device)

        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr,
                                                 relabel_nodes=True, num_nodes=self.news_graph.num_nodes)

        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k] + sum_num_nodes

    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    # if line.strip().split('\t')[3]:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news = self.line_mapper(
                        line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))

                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                        clicked_graphs, mappings, candidates, labels, candidate_entity_list, entity_mask_list = [], [], [], [], [], []
                        sum_num_news = 0

                if (len(clicked_graphs) > 0):
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                    f.seek(0)

class TrainGraphDatasetWithMajorClustering(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, clusters):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors
        self.clusters = clusters

    def line_mapper(self, line, sum_num_news):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # ------------------ Clicked News ----------------------
        # Convert clicked news to indices
        click_idx = self.trans_to_nindex(click_id)

        # Count occurrences of each cluster in clicked news
        cluster_count = {}
        for idx in click_idx:
            for cluster, nodes in self.clusters.items():
                if idx in nodes:
                    if cluster not in cluster_count:
                        cluster_count[cluster] = 0
                    cluster_count[cluster] += 1
                    break

        # Find the majority cluster (cluster with the most clicked news items)
        if cluster_count:
            majority_cluster_id = max(cluster_count, key=cluster_count.get)
        else:
            # Handle cases where no clicked news belongs to any cluster (e.g., assign a default cluster or handle as a special case)
            majority_cluster_id = None  # Or assign some default cluster ID if necessary

        if majority_cluster_id is None:
            # Handle cases where clicked news is not in any cluster (skip or assign default cluster)
            pass
        # Get only neighbors within the same cluster
        top_k = len(click_id)
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                # Filter neighbors by cluster
                neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if n in self.clusters[majority_cluster_id]]
                current_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # Build the subgraph using the filtered click_idx within the same cluster
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)
        padded_maping_idx = F.pad(mapping_idx, (self.cfg.model.his_size - len(mapping_idx), 0), "constant", -1)

        # ------------------ Candidate News ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]
            candidate_neighbor_entity = np.zeros(
                ((self.cfg.npratio + 1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio + 1,
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
            sum_num_news + sub_news_graph.num_nodes
    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset:
            subset = [0]

        subset = torch.tensor(subset, dtype=torch.long, device=device)

        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr,
                                                 relabel_nodes=True, num_nodes=self.news_graph.num_nodes)

        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k] + sum_num_nodes

    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    # if line.strip().split('\t')[3]:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news = self.line_mapper(
                        line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))

                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                        clicked_graphs, mappings, candidates, labels, candidate_entity_list, entity_mask_list = [], [], [], [], [], []
                        sum_num_news = 0

                if (len(clicked_graphs) > 0):
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                    f.seek(0)

class TrainGraphDataseClusterId(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, clusters, cluster_id):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)

        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors
        self.clusters = clusters
        self.cluster_id = cluster_id

    def line_mapper(self, line, sum_num_news):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        # ------------------ Clicked News ----------------------
        # Convert clicked news to indices
        click_idx = self.trans_to_nindex(click_id)

        # Get only neighbors within the same cluster
        top_k = len(click_id)
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                # Filter neighbors by cluster
                neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if n in self.clusters[self.cluster_id]]
                current_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # Build the subgraph using the filtered click_idx within the same cluster
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, top_k, sum_num_news)
        padded_maping_idx = F.pad(mapping_idx, (self.cfg.model.his_size - len(mapping_idx), 0), "constant", -1)

        # ------------------ Candidate News ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]
            candidate_neighbor_entity = np.zeros(
                ((self.cfg.npratio + 1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio + 1,
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return sub_news_graph, padded_maping_idx, candidate_input, candidate_entity, entity_mask, label, \
            sum_num_news + sub_news_graph.num_nodes
    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset:
            subset = [0]

        subset = torch.tensor(subset, dtype=torch.long, device=device)

        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr,
                                                 relabel_nodes=True, num_nodes=self.news_graph.num_nodes)

        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k] + sum_num_nodes

    def __iter__(self):
        while True:
            clicked_graphs = []
            candidates = []
            mappings = []
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    # if line.strip().split('\t')[3]:
                    sub_newsgraph, padded_mapping_idx, candidate_input, candidate_entity, entity_mask, label, sum_num_news = self.line_mapper(
                        line, sum_num_news)

                    clicked_graphs.append(sub_newsgraph)
                    candidates.append(torch.from_numpy(candidate_input))
                    mappings.append(padded_mapping_idx)
                    labels.append(label)

                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))

                    if len(clicked_graphs) == self.batch_size:
                        batch = Batch.from_data_list(clicked_graphs)

                        candidates = torch.stack(candidates)
                        mappings = torch.stack(mappings)
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)

                        labels = torch.tensor(labels, dtype=torch.long)
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                        clicked_graphs, mappings, candidates, labels, candidate_entity_list, entity_mask_list = [], [], [], [], [], []
                        sum_num_news = 0

                if (len(clicked_graphs) > 0):
                    batch = Batch.from_data_list(clicked_graphs)

                    candidates = torch.stack(candidates)
                    mappings = torch.stack(mappings)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                    f.seek(0)


class TrainGraphDatasetClusterIds(TrainDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, clusters, cluster_ids):
        super().__init__(filename, news_index, news_input, local_rank, cfg)
        self.neighbor_dict = neighbor_dict
        self.news_graph = news_graph.to(local_rank, non_blocking=True)
        self.batch_size = cfg.batch_size / cfg.gpu_num
        self.entity_neighbors = entity_neighbors
        self.clusters = clusters
        self.cluster_ids = [0,1,2,3,4]

    def line_mapper(self, line, sum_num_news):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        sess_pos = line[4].split()
        sess_neg = line[5].split()
        sub_graphs = []
        mapping_indexes = []
        # ------------------ Clicked News ----------------------
        # Convert clicked news to indices
        click_idx = self.trans_to_nindex(click_id)

        # Get only neighbors within the same cluster
        top_k = len(click_id)
        source_idx = click_idx
        for cluster_id in self.cluster_ids:
            sub_news_graph = None
            mapping_idx = None

            current_hop_idx = []
            for _ in range(self.cfg.model.k_hops):
                for news_idx in source_idx:
                    # Filter neighbors by cluster
                    if news_idx in self.clusters[cluster_id]:
                        neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if n in self.clusters[cluster_id]]
                        current_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
                        current_hop_idx.append(news_idx)
                    else:
                        continue
                # source_idx = current_hop_idx
                # click_idx.extend(current_hop_idx)
            # Build the subgraph using the filtered click_idx within the same cluster
            sub_news_graph, mapping_idx = self.build_subgraph(current_hop_idx, top_k, sum_num_news)
            padded_maping_idx = torch.clamp(
                F.pad(mapping_idx, (self.cfg.model.his_size - len(mapping_idx), 0), "constant", -1),
                max=sub_news_graph.x.size(0) - 1
            )
            sub_graphs.append(sub_news_graph)
            mapping_indexes.append(padded_maping_idx)
        # ------------------ Candidate News ---------------------
        label = 0
        sample_news = self.trans_to_nindex(sess_pos + sess_neg)
        candidate_input = self.news_input[sample_news]

        # ------------------ Entity Subgraph --------------------
        if self.cfg.model.use_entity:
            origin_entity = candidate_input[:, -3 - self.cfg.model.entity_size:-3]
            candidate_neighbor_entity = np.zeros(
                ((self.cfg.npratio + 1) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(self.cfg.npratio + 1,
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        return sub_graphs, mapping_indexes, candidate_input, candidate_entity, entity_mask, label, \
            sum_num_news + sub_news_graph.num_nodes
    def build_subgraph(self, subset, k, sum_num_nodes):
        device = self.news_graph.x.device

        if not subset:
            subset = [0]

        subset = torch.tensor(subset, dtype=torch.long, device=device)

        unique_subset, unique_mapping = torch.unique(subset, sorted=True, return_inverse=True)
        subemb = self.news_graph.x[unique_subset]
        if unique_mapping.max() >= subemb.size(0):
            print("Warning: unique_mapping contains out-of-bound indices")
            unique_mapping = unique_mapping.clamp(max=subemb.size(0) - 1)

        sub_edge_index, sub_edge_attr = subgraph(unique_subset, self.news_graph.edge_index, self.news_graph.edge_attr,
                                                 relabel_nodes=True, num_nodes=self.news_graph.num_nodes)

        sub_news_graph = Data(x=subemb, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        return sub_news_graph, unique_mapping[:k] + sum_num_nodes

    def __iter__(self):
        while True:
            # Initialize lists for the batch
            clicked_graphs = []  # Will hold lists of subgraphs for each cluster
            candidates = []
            mappings = []  # Will hold lists of mapping indices for each cluster
            labels = []

            candidate_entity_list = []
            entity_mask_list = []
            sum_num_news = 0
            with open(self.filename) as f:
                for line in f:
                    # Call line_mapper to get lists of subgraphs and mapping indices
                    sub_newsgraphs, padded_mapping_idx_list, candidate_input, candidate_entity, entity_mask, label, sum_num_news = self.line_mapper(
                        line, sum_num_news)

                    # sub_newsgraphs is a list of subgraphs (one for each cluster), append them to clicked_graphs
                    clicked_graphs.append(sub_newsgraphs)

                    # Append the padded mapping indices (list of indices for each cluster)
                    # Example: padded_mapping_idx_list should be [[tensor_1, tensor_2, ..., tensor_32], ... (for 5 clusters)]
                    for i in range(len(padded_mapping_idx_list)):
                        # Append the mapping index for each cluster separately
                        if len(mappings) <= i:
                            mappings.append([])  # Ensure mappings has enough lists for each cluster

                        # Append each tensor (for each sample in the batch) to the corresponding cluster list
                        mappings[i].append(padded_mapping_idx_list[i])
                    # Append candidate input, entity data, and label
                    candidates.append(torch.from_numpy(candidate_input))
                    candidate_entity_list.append(torch.from_numpy(candidate_entity))
                    entity_mask_list.append(torch.from_numpy(entity_mask))
                    labels.append(label)

                    # When batch size is reached, process and yield the batch
                    if len(clicked_graphs) == self.batch_size:
                        # We need to process subgraphs for each cluster independently,
                        # since clicked_graphs is now a list of lists of subgraphs (for each cluster)
                        batch = [Batch.from_data_list([clicked_graphs[i][j] for i in range(int(self.batch_size))])
                                 for j in range(len(self.cluster_ids))]

                        # Stack candidates
                        candidates = torch.stack(candidates)

                        # We do not need to stack `mappings` since it is already a list of lists
                        # Stack other data like entities and masks
                        candidate_entity_list = torch.stack(candidate_entity_list)
                        entity_mask_list = torch.stack(entity_mask_list)
                        labels = torch.tensor(labels, dtype=torch.long)

                        # Yield the batch: list of subgraphs (one list for each cluster), list of mapping indices, and other data
                        yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels

                        # Reset for the next batch
                        clicked_graphs, mappings, candidates, labels, candidate_entity_list, entity_mask_list = [], [], [], [], [], []
                        sum_num_news = 0

                # If there are remaining graphs at the end of the file, yield the last batch
                if len(clicked_graphs) > 0:
                    batch = [Batch.from_data_list([clicked_graphs[i][j] for i in range(len(clicked_graphs))])
                             for j in range(len(self.cluster_ids))]

                    candidates = torch.stack(candidates)
                    candidate_entity_list = torch.stack(candidate_entity_list)
                    entity_mask_list = torch.stack(entity_mask_list)
                    labels = torch.tensor(labels, dtype=torch.long)

                    yield batch, mappings, candidates, candidate_entity_list, entity_mask_list, labels
                    f.seek(0)

class ValidGraphDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, news_entity):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]

        click_idx = self.trans_to_nindex(click_id)
        clicked_entity = self.news_entity[click_idx]  
        source_idx = click_idx     
        for _ in range(self.cfg.model.k_hops) :
            current_hop_idx = []
            for news_idx in source_idx:
                current_hop_idx.extend(self.neighbor_dict[news_idx][:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

         # ------------------ Entity --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros((len(candidate_index)*self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt,idx in enumerate(origin_entity.flatten()):
                if idx == 0: continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0: continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]
            
            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index), self.cfg.model.entity_size *self.cfg.model.entity_neighbors)
       
            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels
    
    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels = self.line_mapper(line)
            yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels

class ValidGraphFirstClusterDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, news_entity, clusters):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.clusters = clusters  # Add the clusters dictionary

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]

        # Convert clicked news to indices
        click_idx = self.trans_to_nindex(click_id)

        cluster_id = None
        for idx in click_idx:
            for cluster, nodes in self.clusters.items():
                if idx in nodes:
                    cluster_id = cluster
                    break
            if cluster_id is not None:
                break

        # If no cluster is found for the clicked news, handle as a special case (e.g., skip or assign default cluster)
        if cluster_id is None:
            # Handle cases where clicked news is not in any cluster
            pass


        clicked_entity = self.news_entity[click_idx]

        # Get neighbors that belong to the same cluster as the majority cluster
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                # Filter neighbors based on the majority cluster
                neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if n in self.clusters[cluster_id]]
                current_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # Build the subgraph using the filtered click_idx within the majority cluster
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

        # ------------------ Entity Processing --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros((len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index), self.cfg.model.entity_size * self.cfg.model.entity_neighbors)

            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels

    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels = self.line_mapper(line)


                yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels

class ValidGraphMajorClusterDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, news_entity, clusters):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.clusters = clusters  # Add the clusters dictionary

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]

        # Convert clicked news to indices
        click_idx = self.trans_to_nindex(click_id)

        # Determine the majority cluster based on clicked news items
        cluster_count = {}
        for idx in click_idx:
            for cluster, nodes in self.clusters.items():
                if idx in nodes:
                    if cluster not in cluster_count:
                        cluster_count[cluster] = 0
                    cluster_count[cluster] += 1
                    break

        if cluster_count:
            majority_cluster_id = max(cluster_count, key=cluster_count.get)
        else:
            # Handle cases where no clicked news belongs to any cluster (if needed)
            majority_cluster_id = None

        if majority_cluster_id is None:
            # Handle the case where no valid cluster is found
            pass

        clicked_entity = self.news_entity[click_idx]

        # Get neighbors that belong to the same cluster as the majority cluster
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                # Filter neighbors based on the majority cluster
                neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if n in self.clusters[majority_cluster_id]]
                current_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # Build the subgraph using the filtered click_idx within the majority cluster
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

        # ------------------ Entity Processing --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros((len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index), self.cfg.model.entity_size * self.cfg.model.entity_neighbors)

            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels

    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels = self.line_mapper(line)
                yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels

class ValidGraphClusterIdDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, news_entity, cluster_id, clusters):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.cluster_id = cluster_id
        self.clusters = clusters
        # Add the clusters dictionary

    def line_mapper(self, line):

        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]

        # Convert clicked news to indices
        click_idx = self.trans_to_nindex(click_id)

        # Determine the majority cluster based on clicked news items

        clicked_entity = self.news_entity[click_idx]

        # Get neighbors that belong to the same cluster as the majority cluster
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                # Filter neighbors based on the majority cluster
                neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if n in self.clusters[self.cluster_id]]
                current_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # Build the subgraph using the filtered click_idx within the majority cluster
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

        # ------------------ Entity Processing --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros((len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index), self.cfg.model.entity_size * self.cfg.model.entity_neighbors)

            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels

    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels = self.line_mapper(line)
                yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels


class ValidGraphClusterIdsDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors,
                 news_entity, clusters, cluster_ids):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.cluster_ids = [0, 1, 2, 3, 4]  # Using cluster IDs 0-4 to match training class
        self.clusters = clusters

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]
        click_idx = self.trans_to_nindex(click_id)

        # Initialize lists to hold subgraphs and mapping indices for each cluster
        subgraphs = []
        mapping_indexes = []

        # Process each cluster to generate its subgraph
        for cluster_id in self.cluster_ids:
            source_idx = click_idx
            current_hop_idx = []

            for _ in range(self.cfg.model.k_hops):
                temp_hop_idx = []
                for news_idx in source_idx:
                    # Filter neighbors by cluster
                    if news_idx in self.clusters[cluster_id]:
                        neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if
                                                n in self.clusters[cluster_id]]
                        temp_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
                        temp_hop_idx.append(news_idx)
                # source_idx = temp_hop_idx
                current_hop_idx.extend(temp_hop_idx)

            # Build subgraph for this cluster using filtered neighbors within the cluster
            sub_news_graph, mapping_idx = self.build_subgraph(current_hop_idx, len(click_id), 0)

            # Pad mapping indices to match `his_size` with appropriate bounds
            padded_mapping_idx = torch.clamp(
                F.pad(mapping_idx, (self.cfg.model.his_size - len(mapping_idx), 0), "constant", -1),
                max=sub_news_graph.x.size(0) - 1
            )
            subgraphs.append(sub_news_graph)
            mapping_indexes.append(padded_mapping_idx)

        # Process candidate news and entities
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        # Entity processing if enabled
        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros(
                (len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64
            )
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index),
                                                                          self.cfg.model.entity_size * self.cfg.model.entity_neighbors)

            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1
            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        # Create a batch for each subgraph list across clusters
        batch = [Batch.from_data_list([subgraphs[j]]) for j in range(len(self.cluster_ids))]

        return batch, mapping_indexes, click_idx, candidate_input, candidate_entity, entity_mask, labels

    def __iter__(self):
        mappings = []
        clicked_graphs = []# Will hold lists of mapping indices for each cluster

        with open(self.filename) as f:
            for line in f:
                if line.strip().split('\t')[3]:
                    sub_newsgraphs, padded_mapping_idx_list, candidates, candidate_input, candidate_entity, entity_mask, labels = self.line_mapper(
                        line)
                    clicked_graphs.append(sub_newsgraphs)

                    for i in range(len(padded_mapping_idx_list)):
                        # Append the mapping index for each cluster separately
                        if len(mappings) <= i:
                            mappings.append([])  # Ensure mappings has enough lists for each cluster

                        # Append each tensor (for each sample in the batch) to the corresponding cluster list
                        mappings[i].append(padded_mapping_idx_list[i])
                    batch = [Batch.from_data_list([clicked_graphs[i][j] for i in range(len(clicked_graphs))])
                             for j in range(len(self.cluster_ids))]


                    yield batch, mappings, candidates, candidate_input, candidate_entity, entity_mask, labels


class ValidGraphAllClusterDataset(TrainGraphDataset):
    def __init__(self, filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors, news_entity, clusters, cluster_ids):
        super().__init__(filename, news_index, news_input, local_rank, cfg, neighbor_dict, news_graph, entity_neighbors)
        self.news_graph.x = torch.from_numpy(self.news_input).to(local_rank, non_blocking=True)
        self.news_entity = news_entity
        self.clusters = clusters  # Dictionary of clusters
        self.cluster_ids = cluster_ids  # List of cluster IDs
        self.current_cluster_idx = 0  # Initialize current cluster index

    def line_mapper(self, line):
        line = line.strip().split('\t')
        click_id = line[3].split()[-self.cfg.model.his_size:]

        # Convert clicked news to indices
        click_idx = self.trans_to_nindex(click_id)

        # Get the current cluster to be used
        current_cluster_id = self.cluster_ids[self.current_cluster_idx]

        # Determine the majority cluster based on clicked news items
        clicked_entity = self.news_entity[click_idx]

        # Get neighbors that belong to the same cluster as the current cluster
        source_idx = click_idx
        for _ in range(self.cfg.model.k_hops):
            current_hop_idx = []
            for news_idx in source_idx:
                # Filter neighbors based on the current cluster
                neighbors_in_cluster = [n for n in self.neighbor_dict[news_idx] if n in self.clusters[current_cluster_id]]
                current_hop_idx.extend(neighbors_in_cluster[:self.cfg.model.num_neighbors])
            source_idx = current_hop_idx
            click_idx.extend(current_hop_idx)

        # Build the subgraph using the filtered click_idx within the current cluster
        sub_news_graph, mapping_idx = self.build_subgraph(click_idx, len(click_id), 0)

        # Move to the next cluster for the next batch
        self.current_cluster_idx = (self.current_cluster_idx + 1) % len(self.cluster_ids)

        # ------------------ Entity Processing --------------------
        labels = np.array([int(i.split('-')[1]) for i in line[4].split()])
        candidate_index = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
        candidate_input = self.news_input[candidate_index]

        if self.cfg.model.use_entity:
            origin_entity = self.news_entity[candidate_index]
            candidate_neighbor_entity = np.zeros((len(candidate_index) * self.cfg.model.entity_size, self.cfg.model.entity_neighbors), dtype=np.int64)
            for cnt, idx in enumerate(origin_entity.flatten()):
                if idx == 0:
                    continue
                entity_dict_length = len(self.entity_neighbors[idx])
                if entity_dict_length == 0:
                    continue
                valid_len = min(entity_dict_length, self.cfg.model.entity_neighbors)
                candidate_neighbor_entity[cnt, :valid_len] = self.entity_neighbors[idx][:valid_len]

            candidate_neighbor_entity = candidate_neighbor_entity.reshape(len(candidate_index), self.cfg.model.entity_size * self.cfg.model.entity_neighbors)

            entity_mask = candidate_neighbor_entity.copy()
            entity_mask[entity_mask > 0] = 1

            candidate_entity = np.concatenate((origin_entity, candidate_neighbor_entity), axis=-1)
        else:
            candidate_entity = np.zeros(1)
            entity_mask = np.zeros(1)

        batch = Batch.from_data_list([sub_news_graph])

        return batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels

    def __iter__(self):
        for line in open(self.filename):
            if line.strip().split('\t')[3]:
                batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels = self.line_mapper(line)
                yield batch, mapping_idx, clicked_entity, candidate_input, candidate_entity, entity_mask, labels



class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


