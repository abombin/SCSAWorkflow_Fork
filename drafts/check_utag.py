import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import warnings
import logging
import scanpy.external as sce
from scipy import stats
import umap as umap_lib
from scipy.sparse import issparse
from typing import List, Union, Optional
from numpy.lib import NumpyVersion
import sys

import multiprocessing
import parmap

# Set the start method to 'fork'
multiprocessing.set_start_method('fork', force=True)

#os.chdir("/Users/bombina2/github/SCSAWorkflow_Fork/src/spac")
os.chdir("/Users/bombina2/github/SCSAWorkflow_Fork/drafts")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/spac')))

from utils import check_table, check_annotation, check_feature

import scanpy as sc
from utag_functions import utag
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='SPAC:%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def _validate_transformation_inputs(
        adata: anndata,
        layer: Optional[str] = None,
        associated_table: Optional[str] = None,
        features: Optional[Union[List[str], str]] = None
        ) -> None:
    """
    Validate inputs for transformation functions.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Name of the layer in `adata` to use for transformation.
    associated_table : str, optional
        Name of the key in `obsm` that contains the numpy array.
    features : list of str or str, optional
        Names of features to use for transformation.

    Raises
    ------
    ValueError
        If both `associated_table` and `layer` are specified.
    """

    if associated_table is not None and layer is not None:
        raise ValueError("Cannot specify both"
                         f" 'associated table':'{associated_table}'"
                         f" and 'table':'{layer}'. Please choose one.")

    if associated_table is not None:
        check_table(adata=adata,
                    tables=associated_table,
                    should_exist=True,
                    associated_table=True)
    else:
        check_table(adata=adata,
                    tables=layer)

    if features is not None:
        check_feature(adata, features=features)


def _select_input_features(adata: anndata,
                           layer: str = None,
                           associated_table: str = None,
                           features: Optional[Union[str, List[str]]] = None,

                           ) -> np.ndarray:
    """
    Selects the numpy array to be used as input for transformations

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    layer : str, optional
        Layer of AnnData object for UMAP. Defaults to `None`.
    associated_table : str, optional
        Name of the key in `adata.obsm` that contains the numpy array.
        Defaults to `None`.
    features : str or List[str], optional
        Names of the features to select from layer. If None, all features are
        selected. Defaults to None.

    Returns
    -------
    np.ndarray
        The selected numpy array.

    """
    if associated_table is not None:
        # Flatten the obsm numpy array before returning it
        logger.info(f'Using the associated table:"{associated_table}"')
        np_array = adata.obsm[associated_table]
        return np_array.reshape(np_array.shape[0], -1)
    else:
        np_array = adata.layers[layer] if layer is not None else adata.X
        logger.info(f'Using the table:"{layer}"')
        if features is not None:
            if isinstance(features, str):
                features = [features]

            logger.info(f'Using features:"{features}"')
            np_array = np_array[:,
                                [adata.var_names.get_loc(feature)
                                 for feature in features]]
        return np_array


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
        parallel = False,
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
        parallel=True,
        processes=n_jobs,
        k=k,
    )

    curClusterCol = 'UTAG Label_leiden_' + str(resolution)
    cluster_list = utag_results.obs[curClusterCol].copy()
    adata.obs[output_annotation] = pd.Categorical(cluster_list)
    adata.uns["utag_features"] = features


adata = sc.read("/Users/bombina2/github/multiplex-analysis-web-apps/input/healthy_lung_adata.h5ad")

run_utag_clustering(
        adata,
        features=None,
        k=15,
        resolution=1,
        max_dist=20,
        n_principal_components=10,
        random_state=42,
        n_jobs=5,
        n_iterations=5,
        slide_key="roi",
        layer=None,
        output_annotation="UTAG",
        associated_table=None,
        parallel = True)

print(adata)

print(adata.obs["UTAG"])