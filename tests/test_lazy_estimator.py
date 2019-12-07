import unittest


class TestLazyEstimator(unittest.TestCase):

    def test_lazy_estimator(self):

        from sklearn import svm
        from sklearn.datasets import make_classification
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        from lazygrid.lazy_estimator import LazyPipeline
        from lazygrid.database import load_all_from_db, drop_db
        from lazygrid.plotter import plot_learning_curve
        from lazygrid.statistics import scoring_summary
        from sklearn.model_selection import cross_validate, permutation_test_score
        from sklearn.metrics import plot_confusion_matrix
        import os
        import matplotlib.pyplot as plt
        import pandas as pd

        db_dir = "./db/make_classification/"
        db_name = os.path.join(db_dir, "database.sqlite")
        drop_db(db_name)

        # generate some data to play with
        X, y = make_classification(n_samples=2000, n_informative=5, n_redundant=0, random_state=42)

        anova_filter = SelectKBest(f_regression, k=5)
        clf = svm.SVC(kernel='linear', random_state=42)
        le = LazyPipeline([('anova', anova_filter), ('svc', clf)], database=db_dir)

        s1 = cross_validate(le, X, y, cv=10, return_estimator=True,
                            return_train_score=True, scoring=scoring_summary)
        s2 = cross_validate(le, X, y, cv=10, return_estimator=True,
                            return_train_score=True, scoring=scoring_summary)
        s3 = permutation_test_score(le, X, y, cv=2)

        le.fit(X, y)
        print("Global accuracy: %.4f" % le.score(X, y))

        if not os.path.isdir("./figures/"):
            os.makedirs("./figures/")
        if not os.path.isdir("./results/"):
            os.makedirs("./results/")

        plt.figure()
        plot_confusion_matrix(le, X, y,
                              normalize="all", display_labels=["R", "S"],
                              cmap=plt.cm.Greens)
        plt.tight_layout()
        plt.savefig("./figures/confusion_matrix_norm.png")
        plt.show()

        plt.figure()
        plot_confusion_matrix(le, X, y,
                              display_labels=["R", "S"], cmap=plt.cm.Greens)
        plt.tight_layout()
        plt.savefig("./figures/confusion_matrix_abs.png")
        plt.show()

        plt.figure()
        plot_learning_curve(le, "Learning Curves (Anova + SVC)", X, y, cv=10)
        plt.tight_layout()
        plt.savefig("./figures/learning_curve.png")
        plt.show()

        check = load_all_from_db(db_name)

        results = pd.DataFrame.from_dict(s1)
        results.to_csv("./results/results_summary.csv")

        self.assertEqual(len(check), 260)
        self.assertEqual(len(s1["estimator"]), 10)
        self.assertEqual(len(s2["estimator"]), 10)


suite = unittest.TestLoader().loadTestsFromTestCase(TestLazyEstimator)
unittest.TextTestRunner(verbosity=2).run(suite)
