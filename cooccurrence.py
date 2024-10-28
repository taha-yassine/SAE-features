# %%
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import networkx as nx
import cupy as cp
import cupyx.scipy.sparse as cusparse
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# %%
with safe_open("cached_activations/gpt2-small-res-jb-feature-splitting-TinyStories.safetensors", framework="pt", device=device) as f:
    # 3D indices of form  (batch_id, ctx_pos, feature_id)
    idx = f.get_tensor("locations").to(torch.int)
    # Corresponding activations
    activations = f.get_tensor("activations")

n_features = int(torch.max(idx[:,2])) + 1

# Keep only a subset of samples
# idx = idx[idx[:,0] < 8000]
# Keep only a subset of features
# idx = idx[idx[..., 2] < 1000]

# %%
# Goal: get indices of features activating on the same token for each token (batch index doesn't matter)

# 1. Get unique values of first 2 dims (i.e. absolute token index) and their counts
# Trick is to use Cantor pairing function to have a bijective mapping between (batch_id, ctx_pos) and a unique 1D index
# Faster than running `torch.unique_consecutive` on the first 2 dims
idx_cantor = (idx[:,0] + idx[:,1]) * (idx[:,0] + idx[:,1] + 1) // 2 + idx[:,1]
unique_idx, idx_counts = torch.unique_consecutive(idx_cantor, return_counts=True)
n_tokens = len(unique_idx)

# 2. The Cantor indices are not consecutive, so we create sorted ones from the counts
idx_flat = torch.repeat_interleave(torch.arange(n_tokens, device=device), idx_counts)

# %%
# Plot histogram of feature activation sparsity

plt.figure(figsize=(10, 6))
plt.hist(idx_counts.numpy(force=True), edgecolor='black', bins=50)
plt.title('Histogram of Feature Activation Sparsity')
plt.xlabel('Number of Features Activated')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %%
# Plot histogram of feature activation values

plt.figure(figsize=(10, 6))
logbins = torch.logspace(torch.log10(activations.min()), torch.log10(activations.max()), 50)
plt.hist(activations.numpy(force=True), edgecolor='black', bins=logbins.numpy(force=True))
plt.title('Histogram of Feature Activation Values')
plt.xlabel('Activation Value')
plt.ylabel('Count')
# plt.xscale('log')  # Set x-axis to logarithmic scale
plt.tight_layout()
plt.show()

# %%
# cooc_matrix = cusparse.csr_matrix((n_features, n_features))

# batch_size = n_tokens
# epochs = math.ceil(n_tokens/batch_size)

# slices = torch.cumsum(torch.zeros(epochs+1, dtype=idx_counts.dtype, device=device).scatter_add_(0, torch.repeat_interleave(torch.arange(epochs, device=device), batch_size)[:-(batch_size-n_tokens%batch_size)]+1, idx_counts),0) # TODO: explain this monstrosity

# for i in tqdm(range(epochs), desc="Processing batches", unit="batch"):
#     rows = cp.asarray(idx[slices[i]:slices[i+1], 2])
#     cols = cp.asarray(idx_flat[slices[i]:slices[i+1]])
#     data = cp.ones(slices[i+1] - slices[i])
#     sparse_matrix = cusparse.coo_matrix((data, (rows, cols)), shape=(n_features, n_tokens))
#     gram = sparse_matrix @ sparse_matrix.T
#     cooc_matrix += gram

# Use cupy as it supports sparse matrices better than pytorch
rows = cp.asarray(idx[:, 2])
cols = cp.asarray(idx_flat)
data = cp.ones(len(rows))
sparse_matrix = cusparse.coo_matrix((data, (rows, cols)), shape=(n_features, n_tokens))
cooc_matrix = sparse_matrix @ sparse_matrix.T

# %%
# Compute Jaccard similarity
def compute_jaccard(cooc_matrix):
    self_occurrence = cooc_matrix.diagonal()
    jaccard_matrix = cooc_matrix / (self_occurrence[:, None] + self_occurrence - cooc_matrix)
    return jaccard_matrix

# Compute Jaccard similarity matrix
jaccard_matrix = compute_jaccard(cooc_matrix)

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
threshold = 0.1 # Needs to be adjusted
jaccard_matrix_thresholded = cusparse.coo_matrix(
    (jaccard_matrix.data[jaccard_matrix.data > threshold], (jaccard_matrix.row[jaccard_matrix.data > threshold], jaccard_matrix.col[jaccard_matrix.data > threshold])),
    shape=(n_features, n_features)
)

# Create undirected graph
g = nx.from_scipy_sparse_array(jaccard_matrix_thresholded)

# Remove self-loops
g.remove_edges_from(nx.selfloop_edges(g))

pos = nx.spring_layout(g)
plt.figure(figsize=(10, 6))
nx.draw(g, pos, with_labels=False, node_size=5, node_color='skyblue', edge_color='gray')
plt.title('Feature Co-occurrence Graph')
plt.show()

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