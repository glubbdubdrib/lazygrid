import unittest


class TestPlotter(unittest.TestCase):

    def test_plot_boxplots(self):

        import lazygrid as lg

        score_list = [[0.98, 0.99, 0.95, 0.93], [0.85, 0.88, 0.87, 0.88], [0.94, 0.98, 0.83, 0.88]]
        labels = ["Model 1", "Model 2", "Model 3"]
        file_name = "box_plot_scores.png"
        title = "Model comparison"

        lg.plotter.plot_boxplots(score_list, labels, file_name, title)


suite = unittest.TestLoader().loadTestsFromTestCase(TestPlotter)
unittest.TextTestRunner(verbosity=2).run(suite)
