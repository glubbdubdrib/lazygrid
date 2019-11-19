import unittest


class TestLogger(unittest.TestCase):

    def test_initialize_logging(self):

        import lazygrid as lg

        logger = lg.logger.initialize_logging()
        logger.info("Log something")
        lg.logger.close_logging(logger)


suite = unittest.TestLoader().loadTestsFromTestCase(TestLogger)
unittest.TextTestRunner(verbosity=2).run(suite)
