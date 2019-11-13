import unittest


class TestDatabase(unittest.TestCase):

    def test_drop_db(self):

        import sqlite3
        import os
        import lazygrid as lg

        db_dir = "./database"
        db_name = "lazygrid-test"
        db_path = os.path.join(db_dir, db_name)

        lg.drop_db(db_name=db_name)

        db = sqlite3.connect(db_path)
        cursor = db.cursor()

        stmt = '''SELECT name FROM sqlite_master WHERE type='table' AND name=? '''
        table = cursor.execute(stmt, ("MODEL",)).fetchone()

        self.assertTrue(not table)

        db.close()

        lg.drop_db(db_name="lazygrid-test")


suite = unittest.TestLoader().loadTestsFromTestCase(TestDatabase)
unittest.TextTestRunner(verbosity=2).run(suite)

