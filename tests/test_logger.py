import unittest


class TestLogger(unittest.TestCase):

    def test_initialize_logging(self):

        import lazygrid as lg

        logger = lg.initialize_logging()
        logger.info("Log something")
        lg.close_logging(logger)


suite = unittest.TestLoader().loadTestsFromTestCase(TestLogger)
unittest.TextTestRunner(verbosity=2).run(suite)
