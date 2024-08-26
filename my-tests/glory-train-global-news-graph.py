import collections
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import torch.nn.functional as F
from tqdm import tqdm
import random
import pickle
from collections import Counter
import numpy as np
import torch
import json
import itertools


mode = 'train'
reprocess = False
use_graph_type = 0

data_dir = {'train': "/home/melika/Documents/code/news-recommendation-v1/GLORY/data/MINDSmall/train"}

behavior_path = Path(data_dir['train']) / "behaviors.tsv"
origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))

nltk_target_path = Path(data_dir[mode]) / "nltk_news_graph.pt"

reprocess_flag = False
if nltk_target_path.exists() is False:
    reprocess_flag = True

# if (reprocess_flag == False) and (reprocess == False):
#     print(f"[{mode}] All graphs exist !")
#     exit()

# -----------------------------------------News Graph------------------------------------------------
behavior_path = Path(data_dir['train']) / "behaviors.tsv"
origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))

# ------------------- Build Graph -------------------------------------------------------------------

if mode == 'train':
    edge_list, user_set = [], set()
    num_line = len(open(behavior_path, encoding='utf-8').readlines())
    with open(behavior_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors news to News Graph"):
            line = line.strip().split('\t')

            # check duplicate user
            used_id = line[1]
            if used_id in user_set:
                continue
            else:
                user_set.add(used_id)

            # record cnt & read path
            history = line[3].split()
            if len(history) > 1:
                long_edge = [news_dict[news_id] for news_id in history]
                edge_list.append(long_edge)

    # edge count
    node_feat = nltk_token_news
    target_path = nltk_target_path
    num_nodes = len(news_dict) + 1

    short_edges = []
    for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing news edge list"):
        # Trajectory Graph
        if use_graph_type == 0:
            for i in range(len(edge) - 1):
                short_edges.append((edge[i], edge[i + 1]))
                # short_edges.append((edge[i + 1], edge[i]))
        elif use_graph_type == 1:
            # Co-occurence Graph
            for i in range(len(edge) - 1):
                for j in range(i + 1, len(edge)):
                    short_edges.append((edge[i], edge[j]))
                    short_edges.append((edge[j], edge[i]))
        else:
            assert False, "Wrong"

    edge_weights = Counter(short_edges)
    unique_edges = list(edge_weights.keys())

    edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
    edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

    data = Data(x=torch.from_numpy(node_feat),
                edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=num_nodes)

    torch.save(data, target_path)
    print(data)
    print(f"[{mode}] Finish News Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")
