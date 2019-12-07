import unittest


class TestDatabase(unittest.TestCase):

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

        db_name = "./database/database.sqlite"
        lg.database.drop_db(db_name)

        X, y = make_classification(random_state=42)

        model = lg.lazy_estimator.LazyPipeline([("ridge", RidgeClassifier())])
        results = cross_validate(model, X, y, cv=10)

        db_entries = lg.database.load_all_from_db(db_name)

        self.assertEqual(len(db_entries), 10)


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatabase)
unittest.TextTestRunner(verbosity=2).run(suite)

