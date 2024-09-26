import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from sklearn.cluster import KMeans
from torch_geometric.nn import Node2Vec
import numpy as np

# Load graph data
graph_data = torch.load(
    '/home/melika/Documents/code/news-recommendation-v1/GLORY/data/MINDSmallSample/train/nltk_news_graph.pt')

# Convert PyTorch Geometric data to NetworkX graph
G = to_networkx(graph_data, to_undirected=True)

graph_data = torch.load(
    '/home/melika/Documents/code/news-recommendation-v1/GLORY/data/MINDSmallSample/train/nltk_news_graph.pt')

# Convert PyTorch Geometric data to NetworkX graph
G = to_networkx(graph_data, to_undirected=True)

# Use a standard spring layout to position the nodes
pos = nx.spring_layout(G, seed=42)

# Draw the entire graph without clustering
nx.draw_networkx_nodes(G, pos=pos, node_color='lightblue', node_size=10)
nx.draw_networkx_edges(G, pos=pos)

# Display the graph
plt.tight_layout()
plt.title("Graph Before Clustering")
plt.show()

# ---------------------------------------------CLUSTERING----------------------------------------------------------------

# Perform Node2Vec to get embeddings
node2vec = Node2Vec(graph_data.edge_index, embedding_dim=128, walk_length=10, context_size=5, walks_per_node=10)
embeddings = node2vec(torch.arange(graph_data.num_nodes))
embeddings = embeddings.detach().cpu().numpy()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5)
news_cluster = kmeans.fit_predict(embeddings)

# Group nodes by cluster
clusters = {}
for i, cluster_id in enumerate(news_cluster):
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(i)

# Use the modularity-based clustering approach but with the KMeans clusters
communities = list(clusters.values())

# ------------------------------------------------------VISUALIZE--------------------------------------------------------

# Create a "supergraph" based on clusters
supergraph = nx.cycle_graph(len(communities))

# Compute positions for the node clusters using a larger scale factor
superpos = nx.spring_layout(supergraph, scale=50, seed=429)

# Use the "supernode" positions as the center of each node cluster
centers = list(superpos.values())
pos = {}


# Helper function to apply small random offsets for nodes in the same cluster
def spread_nodes_around_center(center, num_nodes, spread_factor=1.0):
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    radius = np.random.rand(num_nodes) * spread_factor
    offsets = np.c_[np.cos(angles), np.sin(angles)] * radius[:, None]
    return {i: center + offset for i, offset in enumerate(offsets)}


# Position nodes in each community (cluster) around their respective centers
for i, (center, comm) in enumerate(zip(centers, communities)):
    community_pos = spread_nodes_around_center(np.array(center), len(comm), spread_factor=10.0)
    for j, node in enumerate(comm):
        pos[node] = community_pos[j]

# Assign colors to clusters
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Plot nodes for each cluster with distinct colors
for nodes, clr in zip(communities, colors):
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=clr, node_size=10)

# Draw edges
nx.draw_networkx_edges(G, pos=pos)

# Display the graph
plt.tight_layout()
plt.show()
