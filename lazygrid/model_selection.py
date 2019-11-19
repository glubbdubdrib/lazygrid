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

import copy
import traceback
import numpy as np
import pandas as pd
import os
import functools
from typing import Union, Callable, List
from logging import Logger
import pycm
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from .statistics import find_best_solution, confidence_interval_mean_t
from .wrapper import Wrapper
from .plotter import generate_confusion_matrix


def cross_validation(model: Wrapper,
                     x: np.ndarray, y: np.ndarray,
                     x_val: np.ndarray = None, y_val: np.ndarray = None,
                     random_data: bool = True, random_model: bool = True,
                     seed: int = 42, n_splits: int = 10, score_fun: Callable = None,
                     generic_score: Callable = None,
                     logger: Logger = None) -> (dict, list, List[np.ndarray], List[np.ndarray]):
    """
    Apply cross-validation on the given model.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> import lazygrid as lg
    >>>
    >>> x, y = make_classification()
    >>>
    >>> lg_model = lg.wrapper.SklearnWrapper(LogisticRegression())
    >>> score, fitted_models, y_pred_list, y_list = lg.model_selection.cross_validation(model=lg_model, x=x, y=y)
    >>>
    >>> type(score)
    <class 'dict'>
    >>> type(fitted_models)
    <class 'list'>

    Notes
    --------
    The cross-validation can be applied in three different ways, depending on the
    user needs or the application:
        - generating random training and validation sets at each iterations (random_data=True)
        - initializing randomly the given model at each iterations (random_model=True)
        - applying both the previous options

    The second option is usually recommended for deep neural networks and big data sets.

    Parameters
    --------
    :param model: machine learning model
    :param x: input data
    :param y: input labels
    :param x_val: validation data
    :param y_val: validation labels
    :param random_data: if True it enables data randomization
    :param random_model: if True it enables model randomization (if applicable)
    :param seed: seed used to make results reproducible
    :param n_splits: number of cross-validation iterations
    :param score_fun: sklearn-like scoring function used to evaluate the model performance
    :param generic_score: generic score function
    :param logger: object used to save progress
    :return: cross-validation scores, fitted models, list of predicted labels and true labels
    """

    # Check input parameters
    assert score_fun or generic_score or hasattr(model, "score")
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert isinstance(random_data, bool)
    assert isinstance(random_model, bool)
    assert isinstance(seed, int)
    assert isinstance(n_splits, int)
    if not random_data:
        assert x_val is not None and y_val is not None

    if random_model and not random_data:
        assert isinstance(x_val, np.ndarray) and isinstance(y_val, np.ndarray)

    # Useful variables
    fitted_models = []
    y_pred_list = []
    y_list = []
    score = {}

    # prepare data for cross-validation
    if random_data:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        list_of_splits = [split for split in skf.split(x, y)]

    # Cross validation
    if logger: logger.info("Start cross-validation")
    for split_index in range(0, n_splits):

        if logger: logger.info("Split %d" % split_index)

        # randomize data if needed
        if random_data:
            train_index, val_index = list_of_splits[split_index]
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]
        else:
            x_train = x
            y_train = y

        # use generic score function for any kind of purpose
        if generic_score:

            score[split_index] = generic_score(**locals())

        else:

            if not score:
                score = {"train_cv": [], "val_cv": []}

            # load learner
            try:
                learner = copy.deepcopy(model)
            except TypeError:
                learner = model
            learner.set_random_seed(seed, split_index, random_model)

            # check if model has already been computed
            learner.load_model()

            # fit learner
            learner.fit(x_train, y_train)

            # use custom score function for machine learning models having "fit" and "predict" methods
            if score_fun:

                # predict
                y_train_pred = learner.predict(x_train)
                y_val_pred = learner.predict(x_val)

                # compute score
                score_train = score_fun(y_train, y_train_pred)
                score_val = score_fun(y_val, y_val_pred)

                y_pred_list.append(y_val_pred)
                y_list.append(y_val)

            # use default score function provided by the model
            else:
                # compute score directly
                score_train = learner.score(x_train, y_train)
                score_val = learner.score(x_val, y_val)

                y_pred_list.append(learner.predict(x_val))
                y_list.append(y_val)

            # save results
            score["train_cv"].append(score_train)
            score["val_cv"].append(score_val)

            if logger: logger.info("\t%s: train %.4f - validation %.4f" % (str(score_fun), score_train, score_val))

            # save trained model
            learner.save_model()
            fitted_models.append(learner)

        split_index += 1

    return score, fitted_models, y_pred_list, y_list


def _compute_result_summary(models: List[Wrapper], model_id_list: List[List],
                            random_data: bool, random_model: bool,
                            seed: int, n_splits: int, scoring: [Callable, str],
                            test: Callable, alpha: int, cl: float,
                            train_cv: list, val_cv: list,
                            pvalues: list, best_solutions: list) -> pd.DataFrame:
    """
    Compute a summary of the cross-validation and model comparison results.

    Parameters
    --------
    :param models: list of machine learning models (keras or sklearn)
    :param model_id_list: list of model identifiers
    :param random_data: if True it enables data randomization
    :param random_model: if True it enables model randomization (if applicable)
    :param seed: seed used to make results reproducible
    :param n_splits: number of cross-validation iterations
    :param scoring: metric used to evaluate the model performance (f1 or accuracy)
    :param test: statistical test
    :param alpha: significance level
    :param cl: confidence level
    :param train_cv: cross-validation train scores
    :param val_cv: cross-validation validation scores
    :param pvalues: p-values of the statistical hypothesis test
    :param best_solutions: best solutions' indexes
    :return: summary of the results as a table
    """

    columns = [
        "db-name", "db-did",
        "model_name", "model_id", "module", "version", "parameters", "fit_params", "submodels", "is_standalone",
        "train_cv", "val_cv",
        "mean", "ci-l-bound", "ci-u-bound", "separable", "pvalue",
        "test", "alpha", "metric",
        "random-data", "random-model", "seed", "n-splits",
    ]

    results = pd.DataFrame(columns=columns)

    base_row = [
        test.__name__,
        alpha,
        str(scoring),

        random_data,
        random_model,
        seed,
        n_splits,
    ]

    index = 0
    for model in models:
        # compute confidence intervals of the mean of the validation score
        ci_bounds = confidence_interval_mean_t(val_cv[index], cl=cl)

        separable = False if index in best_solutions else True

        try:
            model_names = [str(m.model_name) for m in model.models]
        except:
            model_names = ""

        row = [
            model.dataset_name,
            model.dataset_id,

            model.model_name,
            model_id_list[index],
            model.model_type,
            model.version,
            model.parameters,
            str(model.fit_parameters),
            model_names,
            str(model.is_standalone),
            train_cv[index],
            val_cv[index],
            np.mean(val_cv[index]),
            ci_bounds[0],
            ci_bounds[1],
            separable,
            pvalues[index],
        ]
        row.extend(base_row)

        index += 1

        results = results.append(pd.DataFrame([row], columns=columns), ignore_index=True)

    return results


def _generate_detailed_summary(models: List[Wrapper], confusion_matrix_list: List[pycm.ConfusionMatrix],
                               best_indexes: List[int], output_dir: str) -> None:
    """
    Generate detailed summary about classification results, for the best models only.

    Parameters
    --------
    :param models: list of fitted models
    :param confusion_matrix_list:
    :param best_indexes: indexes of the best models
    :param output_dir: directory where results will be saved
    :return: None
    """

    summary_overall = pd.DataFrame()
    summary_class = pd.DataFrame()

    for i in best_indexes:

        model = models[i]
        confusion_matrix = confusion_matrix_list[i]

        if confusion_matrix:

            if model.model_id:
                model_id = model.model_name + " " + str(model.model_id)
            else:
                model_id = " ".join([m.model_name for m in model.models])

            # overall summary
            prefix_list = []
            for c in confusion_matrix.classes:
                prefix_list.append([model_id, "overall", c])
            prefix_overall = pd.DataFrame(prefix_list, columns=["model", "summary type", "classes"])

            overall_stat = pd.DataFrame.from_dict(confusion_matrix.overall_stat)
            summary = pd.concat([prefix_overall, overall_stat], axis=1).reset_index().drop(columns=["index"])

            summary_overall = pd.concat([summary_overall, summary], ignore_index=True)

            # class summary
            prefix_list = []
            for c in confusion_matrix.classes:
                prefix_list.append([model_id, "class summary", c])
            prefix_class = pd.DataFrame(prefix_list, columns=["model", "summary type", "classes"])

            class_stat = pd.DataFrame.from_dict(confusion_matrix.class_stat).reset_index()
            summary = pd.concat([prefix_class, class_stat], axis=1).reset_index().drop(columns=["index"])

            summary_class = pd.concat([summary_class, summary], ignore_index=True)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    summary_class.to_csv(os.path.join(output_dir, "summary_class.csv"))
    summary_overall.to_csv(os.path.join(output_dir, "summary_overall.csv"))


def compare_models(models: List[Wrapper],
                   x_train: np.ndarray, y_train: np.ndarray,
                   x_val: np.ndarray = None, y_val: np.ndarray = None,
                   random_data: bool = True, random_model: bool = True,
                   seed: int = 42, n_splits: int = 10, score_fun: Callable = None,
                   generic_score: Callable = None,
                   test: Callable = mannwhitneyu, alpha: int = 0.05, cl: float = 0.05,
                   experiment_name: str = "model_comparison", output_dir: str = "./output",
                   verbose: bool = False, logger: Logger = None,
                   class_names: dict = None, font_scale: float = 1,
                   encoding: str = "categorical") -> pd.DataFrame:
    """
    Compare machine learning models' performance on the provided data set, using
    cross-validation and statistical hypothesis tests.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression, RidgeClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> import pandas as pd
    >>> import lazygrid as lg
    >>>
    >>> x, y = make_classification(random_state=42)
    >>>
    >>> lg_model_1 = lg.wrapper.SklearnWrapper(LogisticRegression())
    >>> lg_model_2 = lg.wrapper.SklearnWrapper(RandomForestClassifier())
    >>> lg_model_3 = lg.wrapper.SklearnWrapper(RidgeClassifier())
    >>>
    >>> models = [lg_model_1, lg_model_2, lg_model_3]
    >>> results = lg.model_selection.compare_models(models=models, x_train=x, y_train=y)
    >>>
    >>> pd.set_option('display.width', 9)
    >>> results[['model_name', 'module', 'version', 'ci-l-bound', 'ci-u-bound', 'pvalue']] #doctest: +ELLIPSIS
                   model_name   module version  ci-l-bound  ci-u-bound    pvalue
    0      LogisticRegression  sklearn  0.21.2    0.424323    0.585103  0.000123
    1  RandomForestClassifier  sklearn  0.21.2    0.851488    1.000000  0.829392
    2         RidgeClassifier  sklearn  0.21.2    0.896696    1.000000  1.000000

    Parameters
    --------
    :param models: list of machine learning models (keras or sklearn)
    :param x_train: training data
    :param y_train: input labels
    :param x_val: validation data
    :param y_val: validation labels
    :param random_data: if True it enables data randomization
    :param random_model: if True it enables model randomization (if applicable)
    :param seed: seed used to make results reproducible
    :param n_splits: number of cross-validation iterations
    :param score_fun: sklearn-like scoring function used to evaluate the model performance
    :param generic_score: generic score function
    :param test: statistical test
    :param alpha: significance level
    :param cl: confidence level
    :param experiment_name: name of the current experiment
    :param output_dir: path to the folder where the results will be saved (as csv file)
    :param verbose: if True enables plots
    :param logger: object used to save progress
    :param class_names: dictionary of label names like {0: "Class 1", 1: "Class 2"}
    :param font_scale: font size of figures
    :param encoding: label encoding; accepted values are: "categorical" or "one-hot"
    :return: Pandas DataFrame containing a summary of the results
    """

    confusion_matrix_list = []
    train_cv = []
    val_cv = []
    models_id_list = []

    # Cross-validation
    i = 0
    for model in models:

        score, fitted_models, y_pred_list, y_true_list = cross_validation(model, x=x_train, y=y_train,
                                                                          x_val=x_val, y_val=y_val,
                                                                          random_data=random_data,
                                                                          random_model=random_model,
                                                                          seed=seed, n_splits=n_splits,
                                                                          score_fun=score_fun,
                                                                          generic_score=generic_score,
                                                                          logger=logger)

        conf_mat = generate_confusion_matrix(fitted_models[-1].model_id, fitted_models[-1].model_name,
                                             y_pred_list, y_true_list, class_names,
                                             font_scale, output_dir, encoding)

        confusion_matrix_list.append(conf_mat)
        models_id_list.append([m.model_id for m in fitted_models])
        models[i] = fitted_models[-1]
        train_cv.append(score["train_cv"])
        val_cv.append(score["val_cv"])
        i += 1

    # Find best solution
    best_sol, best_solutions, pvalues = find_best_solution(val_cv,
                                                           test=mannwhitneyu, alpha=alpha,
                                                           use_continuity=False, alternative="two-sided")

    # Compute results' summary
    results = _compute_result_summary(models, models_id_list, random_data, random_model,
                                      seed, n_splits, score_fun, test, alpha, cl,
                                      train_cv, val_cv, pvalues, best_solutions)

    # Save results to csv
    experiment = experiment_name + ".csv"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    results.to_csv(os.path.join(output_dir, experiment))

    # save detailed summary for the best models
    _generate_detailed_summary(models, confusion_matrix_list, best_solutions, output_dir)

    return results
