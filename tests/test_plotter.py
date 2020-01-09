import unittest


class TestPlotter(unittest.TestCase):

    def test_plot_boxplots(self):

        import lazygrid as lg
        import os
        import shutil

        if os.path.isdir("./figures/"):
            shutil.rmtree("./figures/")

        score_list = [[0.98, 0.99, 0.95, 0.93], [0.85, 0.88, 0.87, 0.88], [0.94, 0.98, 0.83, 0.88]]
        labels = ["Model 1", "Model 2", "Model 3"]
        file_name = "box_plot_scores.png"
        title = "Model comparison"

        lg.plotter.plot_boxplots(score_list, labels, file_name, title)

    def test_plot_boxplots_advanced(self):

        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_validate
        import lazygrid as lg
        import pandas as pd

        lg.database.drop_db("./database/database.sqlite")
        X, y = make_classification(random_state=42)
        X = pd.DataFrame(X)

        model1 = lg.lazy_estimator.LazyPipeline([("ridge", RidgeClassifier())])
        model2 = lg.lazy_estimator.LazyPipeline([("logreg", LogisticRegression())])
        model3 = lg.lazy_estimator.LazyPipeline([("ranfor", RandomForestClassifier())])

        models = [model1, model2, model3]

        score_list = []
        labels = []
        for model in models:
            results = cross_validate(model, X, y, cv=10)
            score_list.append(results["test_score"])
            labels.append(model.steps[0][0])

        file_name = "val_scores"
        title = "Model comparison"
        lg.plotter.plot_boxplots(score_list, labels, file_name, title)


suite = unittest.TestLoader().loadTestsFromTestCase(TestPlotter)
unittest.TextTestRunner(verbosity=2).run(suite)
