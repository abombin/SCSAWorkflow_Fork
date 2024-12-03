import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import phenograph_clustering


import scanpy as sc
from spac.utag_functions import utag
import pandas as pd
import numpy as np
from spac.transformations import _validate_transformation_inputs, _select_input_features
from spac.run_utag_reg import run_utag_clustering

class TestUtagClustering(unittest.TestCase):
    def setUp(self):
        # This method is run before each test.
        # It sets up a test case with an AnnData object, a list of features,
        # and a layer name.
        n_cells = 100 
        self.adata = AnnData(np.random.rand(n_cells, 3),
                             var=pd.DataFrame(index=['gene1',
                                                     'gene2',
                                                     'gene3']))
        self.adata.layers['counts'] = np.random.rand(100, 3)

        self.features = ['gene1', 'gene2']
        self.layer = 'counts'

        self.syn_dataset = np.array([
                    np.concatenate(
                            (
                                np.random.normal(100, 1, 500),
                                np.random.normal(10, 1, 500)
                            )
                        ),
                    np.concatenate(
                            (
                                np.random.normal(10, 1, 500),
                                np.random.normal(100, 1, 500)
                            )
                        ),
                ]).reshape(-1, 2)

        self.syn_data = AnnData(
                self.syn_dataset,
                var=pd.DataFrame(index=['gene1',
                                        'gene2'])
                )
        self.syn_data.layers['counts'] = self.syn_dataset

        self.syn_data.obsm['derived_features'] = \
            self.syn_dataset

        # Larger synthetic data for feature subsetting
        self.large_syn_dataset = np.array([
                    np.concatenate(
                            (
                                np.random.normal(100, 1, 500),
                                np.random.normal(10, 1, 500),
                                np.random.normal(10, 1, 500)
                            )
                        ),
                    np.concatenate(
                            (
                                np.random.normal(10, 1, 500),
                                np.random.normal(100, 1, 500),
                                np.random.normal(100, 1, 500)
                            )
                        ),
                    np.concatenate(
                            (
                                np.random.normal(10, 1, 500),
                                np.random.normal(100, 1, 500),
                                np.random.normal(10, 1, 500)
                            )
                        ),
                    np.concatenate(
                            (
                                np.random.normal(10, 1, 500),
                                np.random.normal(10, 1, 500),
                                np.random.normal(100, 1, 500)
                            )
                        )
                ]).T

        self.large_syn_data = AnnData(self.large_syn_dataset,
                                      var=pd.DataFrame(index=['gene1',
                                                              'gene2',
                                                              'gene3',
                                                              'gene4']))
        self.large_syn_data.layers['counts'] = self.large_syn_dataset

        self.large_syn_data.obsm['derived_features'] = \
            self.large_syn_dataset
            
            
    def test_same_cluster_assignments_with_same_seed(self):
        # Run phenograph_clustering with a specific seed
        # and store the cluster assignments
        run_utag_clustering(adata=self.adata, features=self.features, layer=self.layer, random_state=42)
        first_run_clusters = self.adata.obs['UTAG'].copy()

        # Reset the phenograph annotation and run again with the same seed
        del self.adata.obs['UTAG']
        run_utag_clustering(adata=self.adata, features=self.features, layer=self.layer, random_state=42)

        # Check if the cluster assignments are the same
        self.assertTrue(
            (first_run_clusters == self.adata.obs['UTAG']).all()
        )
