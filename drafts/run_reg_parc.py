import parc
import scanpy as sc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='SCSA:%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_parc_clustering(
        adata,
        n_neighbors=30,
        dist_std_local=3,
        jac_std_global='median',
        small_pop=10,
        random_seed=42,
        resolution_parameter=1,
        hnsw_param_ef_construction=150,
        n_principal_components=0,
        n_iterations=5,
        n_jobs=1
):
    """
    Perform PARC clustering on the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    n_neighbors : int
        Number of neighbors to consider for clustering.
    dist_std_local : float
        Standard deviation for local distance.
    jac_std_global : float
        Standard deviation for global distance.
    small_pop : float
        Threshold for small population.
    random_seed : int
        Seed for random number generation.
    resolution_parameter : float
        Resolution parameter for clustering.
    hnsw_param_ef_construction : int
        Parameter for HNSW construction.
    n_principal_components : int
        Number of principal components to use.
    n_iterations : int
        Number of iterations for Leiden algorithm.
    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with PARC clusters stored in `adata.obs['Cluster']`.
    """
    if n_principal_components == 0:
        data = adata.X
    else:
        sc.pp.pca(adata, n_comps=n_principal_components)
        data = adata.obsm['X_pca']

    parc_results = parc.PARC(
        data,
        dist_std_local=dist_std_local,
        jac_std_global=jac_std_global,
        small_pop=small_pop,
        random_seed=random_seed,
        knn=n_neighbors,
        resolution_parameter=resolution_parameter,
        hnsw_param_ef_construction=hnsw_param_ef_construction,
        partition_type="RBConfigurationVP",
        n_iter_leiden=n_iterations,
        num_threads=n_jobs
    )

    parc_results.run_PARC()
    adata.obs['Cluster'] = parc_results.labels.astype(str)

    return adata

