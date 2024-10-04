# %%
import torch
from safetensors import safe_open
import matplotlib.pyplot as plt
import networkx as nx 
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# %%
with safe_open("data/activations.safetensors", framework="pt", device=device) as f:
    # 3D indices of form  (batch_id, ctx_pos, feature_id)
    idx = f.get_tensor("locations").to(torch.int)
    # Corresponding activations
    activations = f.get_tensor("activations").to(torch.int)

# Keep only a subset of samples
idx = idx[idx[:,0] < 8000]
# Keep only a subset of features
idx = idx[idx[..., 2] < 1000]

# %%
# Hypothesis: something is wrong with the feature activations of the first token of each sequence

# Keep only first token
idx_bos, idx_bos_counts = torch.unique(idx[idx[:, 1] == 0, 0], return_counts=True)

print(idx_bos_counts)
# >> tensor([3020, 3020, 3020,  ...,   3020,  3020,   3020])
# Feature activations have an abnomally high sparsity for the first token of each sequence

# %%
# Filter out the first token
idx = idx[idx[:, 1] != 0]

n_features = torch.max(idx[:,2]) + 1

# %%

# Goal: get indices of features activating on the same token for each token (batch index doesn't matter)

# 1. Get unique values of first 2 dims (i.e. absolute token index) and their counts
# Trick is to use Cantor pairing function to have a bijective mapping between (batch_id, ctx_pos) and a unique 1D index
# Faster than running `torch.unique_consecutive` on the first 2 dims
idx_cantor = (idx[:,0] + idx[:,1]) * (idx[:,0] + idx[:,1] + 1) // 2 + idx[:,1]
unique_idx, idx_counts = torch.unique_consecutive(idx_cantor, return_counts=True)

# 2. Split the last dim based on the counts
# Result is a list of tensors of the indices of features that activate on the same token
grouped = torch.split(idx[:, 2], idx_counts.tolist())

# %%

# Plot histogram

plt.figure(figsize=(10, 6))
plt.hist(idx_counts, edgecolor='black')
plt.title('Histogram of Feature Activation Sparsity')
plt.xlabel('Number of Features Activated')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %%
cooccur_matrix = torch.zeros((n_features, n_features), device=device)

for x in grouped:
    # Compute all possible pairs between elements of x
    pairs = torch.combinations(x, r=2, with_replacement=True)
    
    # Update the cooccurrence matrix
    # Only populate the upper triangule
    cooccur_matrix[pairs[:, 0], pairs[:, 1]] += 1

# %%
# Compute Jaccard similarity matrix
jaccard_matrix = torch.zeros((n_features, n_features), device=device)

# Compute diagonal of cooccur_matrix (self-occurrence counts)
self_occurrence = cooccur_matrix.diag()

# Compute Jaccard similarity
for i in range(n_features):
    for j in range(i, n_features):
        intersection = cooccur_matrix[i, j]
        union = self_occurrence[i] + self_occurrence[j] - intersection
        jaccard_matrix[i, j] = jaccard_matrix[j, i] = intersection / union if union > 0 else 0

# %%
# Plot Jaccard similarity matrix
plt.figure(figsize=(10, 6))
plt.imshow(jaccard_matrix, aspect='equal', cmap='viridis')
plt.colorbar(label='Jaccard Similarity')
plt.title('Jaccard Similarity Matrix')
plt.xlabel('Feature ID')
plt.ylabel('Feature ID')
plt.show()

# %%
# Plot histogram of Jaccard similarity values (excluding zeros)
jaccard_values = jaccard_matrix[jaccard_matrix > 0].flatten()

plt.figure(figsize=(10, 6))
plt.hist(jaccard_values, bins=50, edgecolor='black')
plt.title('Histogram of Non-Zero Jaccard Similarity Values')
plt.yscale('log')
plt.xlabel('Jaccard Similarity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



# %%
# Remove edges corresponding to random coocurrences
threshold = 0.1
jaccard_matrix_thresholded = torch.where(jaccard_matrix >= threshold, jaccard_matrix, torch.zeros_like(jaccard_matrix))

# Create undirected graph
# Only the upper triangule is needed
g = nx.from_numpy_array(jaccard_matrix_thresholded.cpu().numpy())

# Remove self-loops
g.remove_edges_from(nx.selfloop_edges(g))

pos = nx.spring_layout(g)
plt.figure(figsize=(10, 6))
nx.draw(g, pos, with_labels=False, node_size=5, node_color='skyblue', edge_color='gray')
plt.title('Feature Co-occurrence Graph')
plt.show()
