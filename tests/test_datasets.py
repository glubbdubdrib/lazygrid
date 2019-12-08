import unittest


class TestDatasets(unittest.TestCase):

    def test_fetch_datasets(self):

        import lazygrid as lg

        datasets = lg.datasets.fetch_datasets(task="regression", max_samples=200, max_features=10)

        datasets = lg.datasets.fetch_datasets(task="classification", min_classes=2,
                                              max_samples=200, max_features=10, update_data=True)

        self.assertEqual(datasets.loc["iris"].version, 45)
        self.assertEqual(datasets.loc["iris"].did, 42098)
        self.assertEqual(datasets.loc["iris"].n_samples, 150)
        self.assertEqual(datasets.loc["iris"].n_features, 4)
        self.assertEqual(datasets.loc["iris"].n_classes, 3)

        datasets = lg.datasets.fetch_datasets(task="regression", max_samples=200, max_features=10, update_data=True)
        datasets = lg.datasets.fetch_datasets(task="random_task", max_samples=200, max_features=10, update_data=True)

    def test_fetch_all_datasets(self):

        import lazygrid as lg

        datasets = lg.datasets.fetch_datasets(task="classification", update_data=True)
        self.assertEqual(len(datasets), 716)

    def test_load_openml_data(self):

        import numpy as np
        import lazygrid as lg

        x, y, n_classes = lg.datasets.load_openml_dataset(data_id=1)
        x, y, n_classes = lg.datasets.load_openml_dataset(data_id=-1)
        x, y, n_classes = lg.datasets.load_openml_dataset(dataset_name="iris")

        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(x.shape, (150, 4))
        self.assertEqual(y.shape, (150,))
        self.assertEqual(n_classes, 3)

    def test_load_npy_dataset(self):

        from sklearn.datasets import make_classification
        import numpy as np
        import lazygrid as lg
        import os

        if not os.path.isdir("./data/"):
            os.makedirs("./data/")

        path_x, path_y = "./data/x.npy", "./data/y.npy"
        if os.path.isfile(path_x):
            os.remove(path_x)
        if os.path.isfile(path_y):
            os.remove(path_y)

        x, y, n_classes = lg.datasets.load_npy_dataset(path_x, path_y)

        self.assertTrue(x is None)
        self.assertTrue(y is None)
        self.assertTrue(n_classes is None)

        x, y = make_classification(random_state=42)
        np.save(path_x, x)
        np.save(path_y, y)

        x, y, n_classes = lg.datasets.load_npy_dataset(path_x, path_y)

        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))
        self.assertEqual(x.shape, (100, 20))
        self.assertEqual(y.shape, (100,))
        self.assertEqual(n_classes, 2)


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasets)
unittest.TextTestRunner(verbosity=2).run(suite)
