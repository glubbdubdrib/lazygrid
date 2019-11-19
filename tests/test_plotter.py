import unittest


class TestPlotter(unittest.TestCase):

    def test_plot_boxplots(self):

        import lazygrid as lg

        score_list = [[0.98, 0.99, 0.95, 0.93], [0.85, 0.88, 0.87, 0.88], [0.94, 0.98, 0.83, 0.88]]
        labels = ["Model 1", "Model 2", "Model 3"]
        file_name = "box_plot_scores.png"
        title = "Model comparison"

        lg.plotter.plot_boxplots(score_list, labels, file_name, title)

    def test_plot_boxplots_advanced(self):

        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import lazygrid as lg

        x, y = make_classification(random_state=42)

        lg_model_1 = lg.wrapper.SklearnWrapper(LogisticRegression())
        lg_model_2 = lg.wrapper.SklearnWrapper(RandomForestClassifier())
        lg_model_3 = lg.wrapper.SklearnWrapper(RidgeClassifier())

        models = [lg_model_1, lg_model_2, lg_model_3]

        score_list = []
        labels = []
        for model in models:
            scores, _, _, _ = lg.model_selection.cross_validation(model, x, y)
            score_list.append(scores["val_cv"])
            labels.append(model.model_name)

        file_name = "val_scores"
        title = "Model comparison"
        lg.plotter.plot_boxplots(score_list, labels, file_name, title)


suite = unittest.TestLoader().loadTestsFromTestCase(TestPlotter)
unittest.TextTestRunner(verbosity=2).run(suite)
