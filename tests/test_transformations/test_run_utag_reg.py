
import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.run_utag_reg import run_utag_clustering

class TestRunUtagClustering(unittest.TestCase):
    def setUp(self):
        # This method is run before each test.
        # It sets up a test case with an AnnData object, a list of features,
        # and a layer name.
        np.random.seed(42)
        n_cells = 100
        self.adata = AnnData(np.random.rand(n_cells, 4),
                             var=pd.DataFrame(index=['gene1', 'gene2', 'gene3', 'gene4']))
        self.adata.layers['counts'] = np.random.rand(n_cells, 4)
        self.adata.obsm['spatial'] = np.random.rand(n_cells, 2)  # Add spatial coordinates
        self.features = ['gene1', 'gene2', 'gene3']
        self.layer = 'counts'
        
        # make a dataset for clustering with 2 clusters
        
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
        
        # add spatial coordinates    
        self.syn_data.obsm['spatial'] = np.array([
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
    


    def test_same_cluster_assignments_with_same_seed(self):
        # Run run_utag_clustering with a specific seed
        # and store the cluster assignments
        run_utag_clustering(adata=self.adata,
                            features=None,
                            k=15,
                            resolution=1,
                            max_dist=20,
                            n_pcs=0,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=None,
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel = False)
        first_run_clusters = self.adata.obs['UTAG'].copy()

        # Reset the UTAG annotation and run again with the same seed
        del self.adata.obs['UTAG']
        run_utag_clustering(adata=self.adata,
                            features=None,
                            k=15,
                            resolution=1,
                            max_dist=20,
                            n_pcs=0,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=None,
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel = False)

        # Check if the cluster assignments are the same
        self.assertTrue(
            (first_run_clusters == self.adata.obs['UTAG']).all()
        )

    def test_typical_case(self):
        # This test checks if the function correctly adds 'UTAG' to the
        # AnnData object's obs attribute and if it correctly sets
        # 'utag_features' in the AnnData object's uns attribute.
        run_utag_clustering(adata=self.adata,
                            features=self.features,
                            k=15,
                            resolution=1,
                            max_dist=20,
                            n_pcs=2,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=self.layer,
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel=False)
        self.assertIn('UTAG', self.adata.obs)
        self.assertEqual(self.adata.uns['utag_features'], self.features)

    def test_output_annotation(self):
        # This test checks if the function correctly adds the "output_annotation" 
        # to the AnnData object's obs attribute 
        output_annotation_name = 'my_output_annotation'
        run_utag_clustering(adata=self.adata,
                            features=self.features,
                            k=15,
                            resolution=1,
                            max_dist=20,
                            n_pcs=2,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=self.layer, 
                            output_annotation=output_annotation_name,
                            associated_table=None,
                            parallel=False)
        self.assertIn(output_annotation_name, self.adata.obs)

    def test_layer_none_case(self):
        # This test checks if the function works correctly when layer is None.
        run_utag_clustering(adata=self.adata,
                            features=self.features,
                            k=15,
                            resolution=1,
                            max_dist=20,
                            n_pcs=2,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=None, 
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel=False)
        self.assertIn('UTAG', self.adata.obs)
        self.assertEqual(self.adata.uns['utag_features'], self.features)

    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer.
        with self.assertRaises(ValueError):
            run_utag_clustering(adata=self.adata,
                                features=self.features,
                                k='invalid',
                                resolution=1,
                                max_dist=20,
                                n_pcs=2,
                                random_state=42,
                                n_jobs=1,
                                n_iterations=5,
                                slide_key=None,
                                layer=self.layer, 
                                output_annotation="UTAG",
                                associated_table=None,
                                parallel=False)
            
    def test_clustering_accuracy(self):
        run_utag_clustering(adata=self.syn_data,
                            features=None,
                            k=15,
                            resolution=1,
                            max_dist=20,
                            n_pcs=0,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=None, 
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel=False)

        self.assertIn('UTAG', self.syn_data.obs)
        self.assertEqual(
            len(np.unique(self.syn_data.obs['UTAG'])),
            2)

if __name__ == '__main__':
    unittest.main()