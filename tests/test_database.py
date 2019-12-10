import unittest


class TestDatabase(unittest.TestCase):

    def test_dir(self):

        import os
        import shutil
        from sklearn import svm
        from sklearn.datasets import make_classification
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        from lazygrid import database, lazy_estimator
        import pandas as pd

        db_name = "./database/database.sqlite"
        db_dir = os.path.dirname(db_name)
        if os.path.isdir(db_dir):
            shutil.rmtree(db_dir)
        database.load_all_from_db(db_name)

        # generate some data to play with
        X, y = make_classification(n_samples=2000, n_informative=5, n_redundant=0, random_state=42)
        X = pd.DataFrame(X)

        anova_filter = SelectKBest(f_regression, k=5)
        clf = svm.SVC(kernel='linear', random_state=42)
        le = lazy_estimator.LazyPipeline([('anova', anova_filter), ('svc', clf)], database=db_dir)

        if os.path.isdir(db_dir):
            shutil.rmtree(db_dir)
        le.fit(X, y)

    def test_drop_db(self):

        import sqlite3
        import lazygrid as lg

        db_name = "./database/lazygrid-test.sqlite"

        lg.database.drop_db(db_name=db_name)

        db = sqlite3.connect(db_name)
        cursor = db.cursor()

        stmt = '''SELECT name FROM sqlite_master WHERE type='table' AND name=? '''
        table = cursor.execute(stmt, ("MODEL",)).fetchone()

        self.assertTrue(not table)

        db.close()

        lg.database.drop_db(db_name=db_name)

    def test_load_all_from_db(self):

        from sklearn.linear_model import RidgeClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_validate
        import lazygrid as lg
        import pandas as pd

        db_name = "./database/database.sqlite"
        lg.database.drop_db(db_name)

        X, y = make_classification(random_state=42)
        X = pd.DataFrame(X)

        model = lg.lazy_estimator.LazyPipeline([("ridge", RidgeClassifier())])
        results = cross_validate(model, X, y, cv=10)

        db_entries = lg.database.load_all_from_db(db_name)

        self.assertEqual(len(db_entries), 10)


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatabase)
unittest.TextTestRunner(verbosity=2).run(suite)

