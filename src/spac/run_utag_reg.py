import scanpy as sc
from spac.utag_functions import utag
import pandas as pd
import numpy as np
from spac.transformations import _validate_transformation_inputs, _select_input_features

def run_utag_clustering(
        adata,
        features=None,
        k=15,
        resolution=1,
        max_dist=20,
        n_principal_components=10,
        random_state=42,
        n_jobs=1,
        n_iterations=5,
        slide_key="Slide",
        layer=None,
        output_annotation="UTAG",
        associated_table=None,
        parallel=False,
        **kwargs
):
    """
    Run UTAG clustering on the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    features : list
        List of features to use for clustering or for PCA. Default (None) is to use all.
    k : int
        The number of nearest neighbor to be used in creating the graph.
        Default is 15.
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
    
    _validate_transformation_inputs(
        adata=adata,
        layer=layer,
        associated_table=associated_table,
        features=features
    )
    
    if not isinstance(k, int) or k <= 0:
        raise ValueError("`k` must be a positive integer")

    if random_state is not None:
        np.random.seed(random_state)

    data = _select_input_features(
        adata=adata,
        layer=layer,
        associated_table=associated_table,
        features=features
    )

    adata_utag = adata.copy()
    adata_utag.X = data
    
    utag_results = utag(
        adata_utag,
        slide_key=slide_key,
        max_dist=max_dist,
        normalization_mode='l1_norm',
        apply_clustering=True,
        clustering_method="leiden",
        resolutions=resolutions,
        leiden_kwargs={"n_iterations": n_iterations, "random_state": random_state},
        pca_kwargs={"n_comps": n_principal_components},
        parallel=parallel,
        processes=n_jobs,
        k=k,
    )

    curClusterCol = 'UTAG Label_leiden_' + str(resolution)
    cluster_list = utag_results.obs[curClusterCol].copy()
    adata.obs[output_annotation] = pd.Categorical(cluster_list)
    adata.uns["utag_features"] = features


