import numpy as np
import pandas as pd
import anndata
#from sklearn.preprocessing import StandardScaler
import scanpy as sc


# Set random seed for reproducibility
np.random.seed(42)

# Number of cells per cluster
n_cells_per_cluster = 50
total_cells = n_cells_per_cluster * 2

# Generate expression values for 4 genes
# Cluster 1: Higher expression in genes 1 and 2
# Cluster 2: Higher expression in genes 3 and 4
cluster1_data = np.random.normal(loc=[8, 7, 2, 1], 
                                scale=[1, 1, 0.5, 0.5], 
                                size=(n_cells_per_cluster, 4))

cluster2_data = np.random.normal(loc=[2, 1, 8, 7], 
                                scale=[0.5, 0.5, 1, 1], 
                                size=(n_cells_per_cluster, 4))

# Combine clusters
expression_matrix = np.vstack([cluster1_data, cluster2_data])

# Create cluster labels
cluster_labels = np.array(['Cluster1'] * n_cells_per_cluster + ['Cluster2'] * n_cells_per_cluster)

# Create cell names and gene names
cell_names = [f'Cell_{i}' for i in range(total_cells)]
gene_names = [f'Gene_{i}' for i in range(4)]

# Create observation (cell) annotations
obs = pd.DataFrame({
    'cluster': cluster_labels,
    'cell_type': ['Type_' + label for label in cluster_labels]
}, index=cell_names)

# Create variable (gene) annotations
var = pd.DataFrame({
    'gene_symbols': gene_names,
    'highly_variable': [True, True, True, True]
}, index=gene_names)

# Create AnnData object
adata = anndata.AnnData(
    X=expression_matrix,
    obs=obs,
    var=var,
    dtype=np.float32
)

adata_2 = adata.copy()

sc.pp.pca(adata)
sc.pp.neighbors(adata, n_pcs=0, random_state=42)

sc.tl.leiden(adata, resolution=1, random_state=42)

sc.tl.umap(adata)
sc.pl.umap(adata, color='leiden')


sc.pp.pca(adata_2)
sc.pp.neighbors(adata_2, n_pcs=2, random_state=42)

sc.tl.leiden(adata_2, resolution=1, random_state=42)

sc.tl.umap(adata_2)
sc.pl.umap(adata_2, color='leiden')