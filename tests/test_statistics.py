import unittest


class TestStatistics(unittest.TestCase):
    
    def test_confidence_interval_mean_t(self):
        
        import numpy as np
        import lazygrid as lg

        np.random.seed(42)
        x = np.random.normal(loc=0, scale=2, size=10)
        confidence_level = 0.05

        l_bound, u_bound = lg.confidence_interval_mean_t(x, confidence_level)

        print(l_bound)
        print(u_bound)

        self.assertAlmostEqual(l_bound, -0.1383, places=4)
        self.assertEqual(u_bound, 1)
        
    def test_find_best_solution(self):
        
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import cross_val_score
        import lazygrid as lg

        x, y = make_classification(random_state=42)

        model1 = LogisticRegression(random_state=42)
        model2 = RandomForestClassifier(random_state=42)
        model3 = RidgeClassifier(random_state=42)
        model_names = ["LogisticRegression", "RandomForestClassifier", "RidgeClassifier"]

        score1 = cross_val_score(estimator=model1, X=x, y=y, cv=10)
        score2 = cross_val_score(estimator=model2, X=x, y=y, cv=10)
        score3 = cross_val_score(estimator=model3, X=x, y=y, cv=10)

        scores = [score1, score2, score3]
        best_idx, best_solutions_idx, pvalues = lg.find_best_solution(scores)

        self.assertEqual(model_names[best_idx], "LogisticRegression")
        self.assertEqual(best_solutions_idx, [0, 2])
        self.assertAlmostEqual(pvalues[0], 0.4783, places=4)
        self.assertAlmostEqual(pvalues[1], 0.0361, places=4)
        self.assertAlmostEqual(pvalues[2], 0.1611, places=4)


suite = unittest.TestLoader().loadTestsFromTestCase(TestStatistics)
unittest.TextTestRunner(verbosity=2).run(suite)
