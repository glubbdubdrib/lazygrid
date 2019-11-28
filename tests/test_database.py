import unittest


class TestDatabase(unittest.TestCase):

    def test_drop_db(self):

        import sqlite3
        import os
        import lazygrid as lg

        db_dir = "./database"
        db_name = "lazygrid-test"
        db_path = os.path.join(db_dir, db_name)

        lg.database.drop_db(db_name=db_name)

        db = sqlite3.connect(db_path)
        cursor = db.cursor()

        stmt = '''SELECT name FROM sqlite_master WHERE type='table' AND name=? '''
        table = cursor.execute(stmt, ("MODEL",)).fetchone()

        self.assertTrue(not table)

        db.close()

        lg.database.drop_db(db_name="lazygrid-test")

    def test_load_all_from_db(self):

        from sklearn.linear_model import RidgeClassifier
        from sklearn.datasets import make_classification
        import lazygrid as lg

        db_name = "lazygrid-db-test"
        lg.database.drop_db(db_name)

        x, y = make_classification(random_state=42)

        model = lg.wrapper.SklearnWrapper(RidgeClassifier(),
                                          db_name=db_name, dataset_id=1,
                                          dataset_name="make-classification")

        lg.model_selection.cross_validation(model, x, y)

        db_entries = lg.database.load_all_from_db(db_name)

        self.assertEqual(len(db_entries), 10)

        for row in db_entries:
            self.assertEqual(row[1], "RidgeClassifier")
            self.assertEqual(row[2], "sklearn")
            self.assertEqual(row[5], "{}")
            self.assertEqual(row[6], "{}")
            self.assertEqual(row[7], "{}")
            self.assertEqual(row[11], 1)
            self.assertEqual(row[12], "make-classification")
            self.assertEqual(row[14], -1)

    def test_load_from_db(self):

        import lazygrid as lg

        db_name = "lazygrid-db-test"
        lg.database.drop_db(db_name)

        create_stmt = '''CREATE TABLE IF NOT EXISTS MY_TABLE(
                             id INTEGER PRIMARY KEY,
                             my_score REAL NOT NULL,
                             my_tag TEXT NOT NULL)'''
        insert_stmt = '''INSERT INTO MY_TABLE(my_score, my_tag) VALUES(?, ?)'''
        query_stmt = '''SELECT * FROM MY_TABLE
                        WHERE my_tag=?'''

        entries = [(2, "tag_1"), (7.5, "tag_2"), (4, "tag_1"), (-121.9, "tag_3")]
        for entry in entries:
            query = [entry[1]]
            lg.database.save_to_db(db_name, entry, query, create_stmt, insert_stmt, query_stmt)

        query = ["tag_1"]

        db_entry = lg.database.load_from_db(db_name, query, create_stmt, query_stmt)

        self.assertEqual(db_entry[0], 1)
        self.assertEqual(db_entry[1], 2.0)
        self.assertEqual(db_entry[2], "tag_1")


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatabase)
unittest.TextTestRunner(verbosity=2).run(suite)

