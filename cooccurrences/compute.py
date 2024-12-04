"""
This script computes feature co-occurrences normalized by their Jaccard similarities from a dataset of feature activations. The computation is done in batches on GPU and saved to disk.
"""

# %%
import torch
from safetensors.torch import save_file
import matplotlib.pyplot as plt
import cupy as cp
import cupyx.scipy.sparse as cusparse
import math
from tqdm import tqdm
from datasets import load_from_disk
import polars as pl
import numpy as np
from itertools import pairwise

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# %%
dataset = load_from_disk("./cached_activations/gpt2-small-res-jb-feature-splitting/rpj-v2-sample")
df = dataset.to_polars()

# %%
n_features = df["feature_idx"].max() + 1
print(f"Number of features: {n_features:,}")

# %%
# Goal: merge batch_idx and ctx_pos into a 1D token_idx
# Easy since batch_idx and ctx_pos are grouped together
# 0 0 -> 0
# 0 0 -> 0
# ...
# 0 1 -> 1
# 0 1 -> 1
# ...
# 1 0 -> i
# 1 0 -> i
# ...

df = df.with_columns(
    (pl.col('batch_idx') != pl.col('batch_idx').shift()) # ┐
    .or_(pl.col('ctx_pos') != pl.col('ctx_pos').shift()) # ┴─ Mark positions where a new token starts
    .fill_null(False) # Fix for first row being null
    .cum_sum() # Increment for each new token (ie. position marked with True)
    .alias('token_idx')
)
n_tokens = df["token_idx"].max() + 1
print(f"Number of tokens: {n_tokens:,}")

# %%
# Plot histogram of feature activation sparsity

plt.figure(figsize=(10, 6))
plt.hist(df.group_by('token_idx').len()['len'], bins=50, edgecolor='black')
plt.title('Histogram of Feature Activation Sparsity')
plt.xlabel('Number of Features Activated')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# %%
# Plot histogram of feature activation values

plt.figure(figsize=(10, 6))
logbins = np.logspace(np.log10(df["activation"].min()), np.log10(df["activation"].max()), 50)
plt.hist(df["activation"], edgecolor='black', bins=logbins)
plt.title('Histogram of Feature Activation Values')
plt.xlabel('Activation Value')
plt.ylabel('Count')
# plt.xscale('log')  # Set x-axis to logarithmic scale
plt.tight_layout()
plt.show()

# %%
cooc_matrix = cusparse.csr_matrix((n_features, n_features))

# Number of tokens per batch
# Can be as high as the GPU can handle
batch_size = min(2**15, n_tokens)

class BatchedDataset:
    def __init__(self, df, batch_size, n_tokens):
        self.df = df
        self.batch_size = batch_size
        self.n_tokens = n_tokens

    def __iter__(self):
        n_batches = math.ceil(self.n_tokens/self.batch_size)
        for start_idx, end_idx in pairwise(range(0, n_batches*self.batch_size, self.batch_size)):
            yield self.df.filter(pl.col('token_idx').is_between(start_idx, end_idx, closed="left"))

    def __len__(self):
        return math.ceil(self.n_tokens/self.batch_size)

for batch in tqdm(BatchedDataset(df, batch_size, n_tokens), desc="Processing batches", unit="batch"):
    rows = cp.asarray(batch["feature_idx"])
    cols = cp.asarray(batch["token_idx"])
    data = cp.ones(len(rows))
    sparse_matrix = cusparse.coo_matrix((data, (rows, cols)), shape=(n_features, n_tokens))
    gram = sparse_matrix @ sparse_matrix.T
    cooc_matrix += gram

# Matrix is symmetric, only keep upper triangle
cooc_matrix = cusparse.triu(cooc_matrix)

# %%
# Compute Jaccard similarity
def compute_jaccard(cooc_matrix):
    self_occurrence = cooc_matrix.diagonal()
    jaccard_matrix = cooc_matrix / (self_occurrence[:, None] + self_occurrence - cooc_matrix)

    # Remove self-loops and only keep upper triangle
    return cusparse.triu(jaccard_matrix, k=1)

# Compute Jaccard similarity matrix
jaccard_matrix = compute_jaccard(cooc_matrix)

# %%
# Save Jaccard similarity matrix to file
tensors_dict = {
    "data": torch.from_numpy(jaccard_matrix.data.get()),
    "row": torch.from_numpy(jaccard_matrix.row.get()),
    "col": torch.from_numpy(jaccard_matrix.col.get()),
    "shape": torch.tensor(jaccard_matrix.shape)
}

save_file(tensors_dict, "jaccard_matrix.safetensors")
