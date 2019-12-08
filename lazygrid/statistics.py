# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero and Giovanni Squillero
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import Callable, List
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import make_scorer, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix


def confidence_interval_mean_t(x: np.ndarray, cl: float = 0.05) -> List:
    """
    Compute the confidence interval of the mean from sample data.

    Parameters
    ----------
    x
        Sample
    cl
        Confidence level

    Returns
    -------
    List
        confidence interval

    Examples
    --------
    >>> import numpy as np
    >>> import lazygrid as lg
    >>>
    >>> np.random.seed(42)
    >>> x = np.random.normal(loc=0, scale=2, size=10)
    >>> confidence_level = 0.05
    >>>
    >>> lg.statistics.confidence_interval_mean_t(x, confidence_level)
    [-0.13829578539063092, 1]


    Notes
    -----
    You should use the t distribution rather than the normal distribution
    when the variance is not known and has to be estimated from sample data.

    When the sample size is large, say 100 or above, the t distribution
    is very similar to the standard normal distribution.
    However, with smaller sample sizes, the t distribution is leptokurtic,
    which means it has relatively more scores in its tails than does the normal distribution.
    As a result, you have to extend farther from the mean to contain a given proportion of the area.
    """
    if np.all(x == np.mean(x)):
        return [np.mean(x), np.mean(x)]
    bounds = stats.t.interval(1-cl, len(x)-1, loc=np.mean(x), scale=stats.sem(x))
    adjusted_bounds = [bound if bound <= 1 else 1 for bound in bounds]
    return adjusted_bounds


def find_best_solution(solutions: list,
                       test: Callable = mannwhitneyu,
                       alpha: float = 0.05,
                       **kwargs) -> (int, list, list):
    """
    Find the best solution in a list of candidates, according to
    a statistical test and a significance level (alpha).

    The best solution is defined as the one having the highest mean value.

    Parameters
    ----------
    solutions
        List of candidate solutions
    test
        Statistical test
    alpha
        Significance level
    kwargs
        Keyword arguments required by the statistical test

    Returns
    -------
    Tuple
        - the position of the best solution inside the candidate input list;
        - the positions of the solutions which are not separable from the best one;
        - the list of p-values returned by the statistical test while comparing the best solution to the other candidates

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression, RidgeClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import cross_val_score
    >>> import lazygrid as lg
    >>>
    >>> x, y = make_classification(random_state=42)
    >>>
    >>> model1 = LogisticRegression(random_state=42)
    >>> model2 = RandomForestClassifier(random_state=42)
    >>> model3 = RidgeClassifier(random_state=42)
    >>> model_names = ["LogisticRegression", "RandomForestClassifier", "RidgeClassifier"]
    >>>
    >>> score1 = cross_val_score(estimator=model1, X=x, y=y, cv=10)
    >>> score2 = cross_val_score(estimator=model2, X=x, y=y, cv=10)
    >>> score3 = cross_val_score(estimator=model3, X=x, y=y, cv=10)
    >>>
    >>> scores = [score1, score2, score3]
    >>> best_idx, best_solutions_idx, pvalues = lg.statistics.find_best_solution(scores)
    >>> model_names[best_idx]
    'LogisticRegression'
    >>> best_solutions_idx
    [0, 2]
    >>> pvalues #doctest: +ELLIPSIS
    [0.4782..., 0.0360..., 0.1610...]
    """

    best_idx = 0
    best_solution = solutions[best_idx]
    best_mean = np.mean(best_solution)

    # find the best solution (the one having the highest mean value)
    index = 1
    for solution in solutions[index:]:
        solution_mean = np.mean(solution)

        if solution_mean > best_mean:
            best_solution = solution
            best_mean = np.mean(best_solution)
            best_idx = index

        index += 1

    best_solutions_idx = []
    pvalues = []

    # check if there are other candidates which may be equivalent to the best one
    index = 0
    for solution in solutions:
        try:
            statistic, pvalue = test(best_solution, solution, **kwargs)
        except ValueError:
            statistic, pvalue = np.inf, np.inf

        if pvalue > alpha:
            best_solutions_idx.append(index)
        pvalues.append(pvalue)

        index += 1

    return best_idx, best_solutions_idx, pvalues


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]


def specificity(y_true, y_pred):
    return tn(y_true, y_pred) / (tn(y_true, y_pred) + fp(y_true, y_pred))


def sensitivity(y_true, y_pred):
    return tp(y_true, y_pred) / (tp(y_true, y_pred) + fn(y_true, y_pred))


scoring_summary = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
                   'fp': make_scorer(fp), 'fn': make_scorer(fn),
                   'specificity': make_scorer(specificity), 'sensitivity': make_scorer(sensitivity),
                   'recall': make_scorer(recall_score), 'accuracy': make_scorer(accuracy_score),
                   'f1': make_scorer(f1_score), 'roc_auc': make_scorer(roc_auc_score)}
