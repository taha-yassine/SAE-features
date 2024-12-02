"""
This script analyses the feature co-occurrence graph.
"""

# %%
import torch
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import networkx as nx
import cupy as cp
import cupyx.scipy.sparse as cusparse
import plotly.graph_objects as go
import json

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
# Load Jaccard similarity matrix
tensors_dict = load_file("./cached_activations/jaccard_matrix.safetensors", device=device)
n_features = tensors_dict["shape"][0].item()
jaccard_matrix = cusparse.coo_matrix((cp.asarray(tensors_dict["data"]), (cp.asarray(tensors_dict["row"]), cp.asarray(tensors_dict["col"]))), shape=(n_features, n_features))

# %%
# Plot Jaccard similarity matrix
plt.figure(figsize=(10, 6))
plt.imshow(jaccard_matrix.get().toarray(), aspect='equal', cmap='viridis')
plt.colorbar(label='Jaccard Similarity')
plt.title('Jaccard Similarity Matrix')
plt.xlabel('Feature ID')
plt.ylabel('Feature ID')
plt.show()

# %%
# Plot histogram of Jaccard similarity values (excluding zeros)
plt.figure(figsize=(10, 6))
plt.hist(jaccard_matrix.data.get(), bins=50, edgecolor='black')
plt.title('Histogram of Non-Zero Jaccard Similarity Values')
# plt.yscale('log')
plt.xlabel('Jaccard Similarity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %%
# Remove edges corresponding to random coocurrences
threshold = 0.2 # Needs to be adjusted

# Filter out self-loops and threshold
mask = (jaccard_matrix.data > threshold) & (jaccard_matrix.row != jaccard_matrix.col)
jaccard_matrix_thresholded = cusparse.coo_matrix(
    (jaccard_matrix.data[mask], (jaccard_matrix.row[mask], jaccard_matrix.col[mask])),
    shape=(n_features, n_features)
)

# Create undirected graph and remove isolated nodes
g = nx.from_scipy_sparse_array(jaccard_matrix_thresholded)
g.remove_nodes_from(list(nx.isolates(g)))

pos = nx.spring_layout(g)

edge_x = []
edge_y = []
for edge in g.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

with open('./cached_activations/exp.json', 'r', encoding='utf-8') as f:
    explanations = json.load(f)

node_x = []
node_y = []
node_text = []
node_colors = []
for node in g.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    try:
        # Node has explanation; show in blue
        node_text.append(f'Feature {node}<br>Degree: {g.degree(node)}<br>{explanations[str(node)]}')
        node_colors.append('#6495ED') 
    except KeyError:
        # Node has no explanation; show in red
        node_text.append(f'Feature {node}<br>Degree: {g.degree(node)}')
        node_colors.append('#ff7f7f')

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    hoverinfo='text',
    text=[str(node) for node in g.nodes()],
    textposition="top center",
    hovertext=node_text,
    marker=dict(
        color=node_colors,
        size=5,
        line_width=1))

fig = go.Figure(data=[edge_trace, node_trace],
               layout=go.Layout(
                   title='Feature Co-occurrence Graph',
                   showlegend=False,
                   hovermode='closest',
                   margin=dict(b=20,l=5,r=5,t=40),
                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
               )

fig.show()

# %%
# Calculate node degrees
degrees = [d for n, d in g.degree()]

# Plot histogram of node degrees
plt.figure(figsize=(10, 6))
plt.hist(degrees, bins=50, edgecolor='black')
plt.title('Histogram of Node Degrees')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.yscale('log')  # Using log scale for better visualization
plt.tight_layout()
plt.show()

