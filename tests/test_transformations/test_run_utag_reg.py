
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
        
        # maka adata for testing clustering based on features vs PCA
        n_cells_complex = 500
        # Generate spatial coordinates in a circular pattern
        theta = np.random.uniform(0, 2*np.pi, n_cells_complex)
        r = np.random.uniform(0, 10, n_cells_complex)
        x_coord = r * np.cos(theta)
        y_coord = r * np.sin(theta)
        # Radial distance-dependent genes (higher expression at the periphery)
        gene1 = np.exp(r/5) + np.random.normal(0, 0.5, n_cells_complex)
        gene2 = -np.exp(r/5) + np.random.normal(0, 0.5, n_cells_complex)
        # Angular position-dependent genes (periodic pattern)
        gene3 = np.sin(3*theta) + np.random.normal(0, 0.3, n_cells_complex)
        gene4 = np.cos(3*theta) + np.random.normal(0, 0.3, n_cells_complex)
        # Quadrant-specific genes
        quadrant = np.where((x_coord > 0) & (y_coord > 0), 1,
                        np.where((x_coord < 0) & (y_coord > 0), 2,
                                np.where((x_coord < 0) & (y_coord < 0), 3, 4)))
        gene5 = np.where(np.isin(quadrant, [1, 2]), 3, 0) + np.random.normal(0, 0.3, n_cells_complex)
        gene6 = np.where(np.isin(quadrant, [1, 4]), 3, 0) + np.random.normal(0, 0.3, n_cells_complex)
        # Random noise genes
        gene7 = np.random.normal(0, 1, n_cells_complex)
        gene8 = np.random.normal(0, 1, n_cells_complex)
        # Combine all genes
        expression_matrix = np.column_stack([gene1, gene2, gene3, gene4, gene5, gene6, gene7, gene8])

        # Create AnnData object
        self.adata_complex = AnnData(
            X=expression_matrix,
            obs=pd.DataFrame(
                {
                    'spatial_x': x_coord,
                    'spatial_y': y_coord,
                    'quadrant': quadrant
                },
                index=[f'cell_{i}' for i in range(n_cells_complex)]
            ),
            var=pd.DataFrame(
                {
                    'gene_type': ['radial', 'radial', 'angular', 'angular', 
                                'quadrant', 'quadrant', 'random', 'random']
                },
                index=[f'gene_{i}' for i in range(8)]
            )
        )

        # Add raw counts layer (here our main matrix is already "normalized")
        self.adata_complex.layers['counts'] = expression_matrix.copy()
        self.adata_complex.obsm["spatial"] = np.random.rand(n_cells_complex, n_cells_complex)
    


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
    
    def test_features_vs_pca_utag_clustering(self):
        run_utag_clustering(
            self.adata_complex,
            features=None,
            k=15,
            resolution=1,
            max_dist=2,
            n_pcs=0,
            random_state=42,
            n_jobs=1,
            n_iterations=5,
            slide_key=None,
            layer=None,
            output_annotation="UTAG",
            associated_table=None,
            parallel = False
            )
        run1_clusters = list(self.adata_complex.obs["UTAG"].copy())

        # Reset the UTAG annotation and run again with the same seed
        del self.adata_complex.obs['UTAG']
        run_utag_clustering(
            self.adata_complex,
            features=None,
            k=15,
            resolution=1,
            max_dist=2,
            n_pcs=2,
            random_state=42,
            n_jobs=1,
            n_iterations=5,
            slide_key=None,
            layer=None,
            output_annotation="UTAG",
            associated_table=None,
            parallel = False
            )
        run2_clusters = list(self.adata_complex.obs["UTAG"].copy())

        # Check if the cluster assignments are different
        self.assertFalse(run1_clusters == run2_clusters)

if __name__ == '__main__':
    unittest.main()