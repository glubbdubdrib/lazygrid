import unittest


class TestLogger(unittest.TestCase):

    def test_initialize_logging(self):

        import lazygrid as lg

        logger = lg.file_logger.initialize_logging()
        logger.info("Log something")
        lg.file_logger.close_logging(logger)


suite = unittest.TestLoader().loadTestsFromTestCase(TestLogger)
unittest.TextTestRunner(verbosity=2).run(suite)
