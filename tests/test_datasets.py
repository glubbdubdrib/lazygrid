import unittest


class TestDatasets(unittest.TestCase):

    def test_fetch_datasets(self):

        import lazygrid as lg

        datasets = lg.fetch_datasets(task="classification", min_classes=2, max_samples=1000, max_features=10)

        self.assertEqual(datasets.loc["iris"].version, 45)
        self.assertEqual(datasets.loc["iris"].did, 42098)
        self.assertEqual(datasets.loc["iris"].n_samples, 150)
        self.assertEqual(datasets.loc["iris"].n_features, 4)
        self.assertEqual(datasets.loc["iris"].n_classes, 3)

    def test_load_openml_data(self):

        import numpy as np
        import lazygrid as lg

        x, y, n_classes = lg.load_openml_dataset(dataset_name="iris")

        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(x.shape, (150, 4))
        self.assertEqual(y.shape, (150,))
        self.assertEqual(n_classes, 3)


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasets)
unittest.TextTestRunner(verbosity=2).run(suite)