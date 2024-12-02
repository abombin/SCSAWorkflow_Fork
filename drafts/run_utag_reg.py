import scanpy as sc
from utag import utag

def run_utag_clustering(
        adata,
        features=None,
        n_neighbors=30,
        resolution=1,
        max_dist=20,
        n_principal_components=10,
        random_state=42,
        n_jobs=7,
        n_iterations=5,
        slide_key="Slide",
        **kwargs
):
    """
    Run UTAG clustering on the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    features : list
        List of features to use for clustering. Default (None) is to use all.
    n_neighbors : int
        The number of nearest neighbor to be used in creating the graph.
        Default is 30.
    resolution : float
        Resolution parameter for the clustering, higher resolution produces 
        more clusters. Default is 1.
    max_dist : float
        Maximum distance to cut edges within a graph. Default is 20.
    n_principal_components : int
        Number of principal components to use for clustering.
    random_state : int
        Random state for reproducibility.
    n_jobs : int
        Number of jobs to run in parallel. Default is 5.
    n_iterations : int
        Number of iterations for the clustering.
    slide_key: str
        Key of adata.obs containing information on the batch structure of the data.
        In general, for image data this will often be a variable indicating the image
        so image-specific effects are removed from data.
        Default is "Slide".

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with clustering results.
    """
    resolutions = [resolution]

    utag_results = utag(
        adata,
        slide_key="Image ID_(standardized)",
        max_dist=max_dist,
        normalization_mode='l1_norm',
        apply_clustering=True,
        clustering_method="leiden",
        resolutions=resolutions,
        leiden_kwargs={"n_iterations": n_iterations, "random_state": random_state},
        pca_kwargs={"n_comps": n_principal_components},
        parallel=True,
        processes=n_jobs
    )

    curClusterCol = 'UTAG Label_leiden_' + str(resolution)
    utag_results.obs['Cluster'] = utag_results.obs[curClusterCol].copy()

    cluster_list = list(utag_results.obs['Cluster'])
    adata.obsp["distances"] = utag_results.obsp["distances"].copy()
    adata.obsp["connectivities"] = utag_results.obsp["connectivities"].copy()
    adata.obsm["X_pca"] = utag_results.obsm["X_pca"].copy()
    adata.uns["neighbors"] = utag_results.uns["neighbors"].copy()
    adata.varm["PCs"] = utag_results.varm["PCs"].copy()
    adata.obs["Cluster"] = cluster_list

    return adata