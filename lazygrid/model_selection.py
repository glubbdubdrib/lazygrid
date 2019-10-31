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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import functools
from typing import Union, Callable, List
from abc import ABCMeta
from logging import Logger
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from keras import Sequential, Model

from .utils import set_random_seed, is_fitted
from .database import save_model, load_model, drop_db
from .statistics import find_best_solution, confidence_interval_mean_t
from .wrapper import ModelWrapper


def _get_learner(model: ModelWrapper,
                 db_name: str, dataset_id: int, dataset_name: str,
                 random_model: bool, split_index: int,
                 seed: int, fit_params: dict) -> ModelWrapper:
    """
    Fetch fitted model from database or return the model itself.

    :param model: machine learning model
    :param db_name: database name
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param random_model: if True it enables model randomization (if applicable)
    :param split_index: cross-validation index
    :param seed: seed used to make results reproducible
    :return: fitted model if possible, otherwise the model itself
    """

    model = copy.deepcopy(model)
    model = set_random_seed(model, random_model, split_index, seed)

    # wrap model
    learner = ModelWrapper(model, fit_params)

    # check if model has already been computed
    learner = load_model(learner, split_index, dataset_id, dataset_name, db_name)

    return learner


def fit_pipeline(learner: ModelWrapper,
                 x_train: np.ndarray, y_train: np.ndarray):
    """
    Fit pipeline if it has not been fitted yet.

    :param learner: machine learning model
    :param x_train: train data
    :param y_train: train labels
    :return: fitted model
    """

    # Learning with Sklearn Pipeline
    assert learner.model_name is "Pipeline"

    # fit steps only if they are not already fitted
    x_train_t = x_train
    i = 0
    for pipeline_step, model in zip(learner.model.steps, learner.models):

        if not is_fitted(model, x_train_t):
            pipeline_step[1].fit(x_train_t, y_train)
            learner.models[i] = ModelWrapper(copy.deepcopy(pipeline_step[1]), is_standalone=False)

        if hasattr(pipeline_step[1], "transform"):
            x_train_t = pipeline_step[1].transform(x_train_t)

        i += 1

    return learner


def cross_validation(model: Union[Sequential, Model, ABCMeta, Pipeline],
                     x: np.ndarray, y: np.ndarray,
                     db_name: str, dataset_id: int, dataset_name: str,
                     x_val: np.ndarray = None, y_val: np.ndarray = None,
                     random_data: bool = True, random_model: bool = True,
                     seed: int = 42, n_splits: int = 10, scoring: Union[Callable, str] = None,
                     logger: Logger = None, fit_params: dict = {}) -> (dict, list):

    """
    Apply cross-validation on the given model.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>>
    >>> x, y = make_classification()
    >>>
    >>> model = LogisticRegression()
    >>> fit_params = {}
    >>>
    >>> score, fitted_models = cross_validation(model=model, x=x, y=y, db_name="database",
    ...                                         dataset_id=1, dataset_name="make-class")
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
    :param db_name: database name
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param x_val: validation data
    :param y_val: validation labels
    :param random_data: if True it enables data randomization
    :param random_model: if True it enables model randomization (if applicable)
    :param seed: seed used to make results reproducible
    :param n_splits: number of cross-validation iterations
    :param scoring: scoring function used to evaluate the model performance (Callable, f1 or accuracy)
    :param logger: object used to save progress
    :param fit_params: arguments used to specify fit parameters of the model
    :return: cross-validation scores and fitted models
    """

    # Check input parameters
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
    score = {"train_blind": [], "test_blind": [], "train_cv": [], "val_cv": []}
    if isinstance(scoring, Callable):
        score_fun = scoring
    elif scoring == "accuracy":
        score_fun = accuracy_score
    elif scoring == "f1":
        score_fun = functools.partial(f1_score, average="weighted")
    else:
        score_fun = None

    # prepare data for cross-validation
    if random_data:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        list_of_splits = [split for split in skf.split(x, y)]
    if random_model and not random_data:
        x_train = x
        y_train = y

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

        # load learner
        learner = _get_learner(model, db_name, dataset_id, dataset_name, random_model, split_index, seed, fit_params)

        # fit learner
        if learner.model_name is "Pipeline":
            learner = fit_pipeline(learner, x_train, y_train)
        else:
            if not is_fitted(learner, x_train):
                learner.model.fit(x_train, y_train, **fit_params)

        if score_fun:
            # predict
            y_train_pred = learner.model.predict(x_train)
            y_val_pred = learner.model.predict(x_val)

            # compute score
            score_train = score_fun(y_train, y_train_pred)
            score_val = score_fun(y_val, y_val_pred)

        elif hasattr(learner.model, "score"):
            # compute score directly
            score_train = learner.model.score(x_train, y_train)
            score_val = learner.model.score(x_val, y_val)

        # save results
        score["train_cv"].append(score_train)
        score["val_cv"].append(score_val)

        # save trained model
        save_model(learner, split_index, dataset_id, dataset_name, db_name)
        fitted_models.append(copy.deepcopy(learner))

        if logger: logger.info("\t%s: train %.4f - validation %.4f" % (str(scoring), score_train, score_val))

        split_index += 1

    return score, fitted_models


def _compute_result_summary(models: List[ModelWrapper], random_data: bool, random_model: bool,
                            seed: int, n_splits: int, scoring: [Callable, str],
                            test: Callable, alpha: int, cl: float,
                            dataset_id: int, dataset_name: str,
                            train_cv: list, val_cv: list,
                            pvalues: list, best_solutions: list) -> pd.DataFrame:
    """
    Compute a summary of the cross-validation and model comparison results.

    :param models: list of machine learning models (keras or sklearn)
    :param random_data: if True it enables data randomization
    :param random_model: if True it enables model randomization (if applicable)
    :param seed: seed used to make results reproducible
    :param n_splits: number of cross-validation iterations
    :param scoring: metric used to evaluate the model performance (f1 or accuracy)
    :param test: statistical test
    :param alpha: significance level
    :param cl: confidence level
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param train_cv: cross-validation train scores
    :param val_cv: cross-validation validation scores
    :param pvalues: p-values of the statistical hypothesis test
    :param best_solutions: best solutions' indexes
    :return: summary of the results as a table
    """

    columns = [
        "db-name", "db-did",
        "model_name", "module", "version", "parameters", "fit_params", "submodels", "is_standalone",
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

        row = [
            dataset_name,
            dataset_id,

            model.model_name,
            model.model_type,
            model.version,
            model.parameters,
            str(model.fit_parameters),
            str(model.models_id),
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


def compare_models(models: list,
                   x_train: np.ndarray, y_train: np.ndarray, params: list,
                   x_val: np.ndarray = None, y_val: np.ndarray = None,
                   random_data: bool = True, random_model: bool = True,
                   seed: int = 42, n_splits: int = 10, scoring: [Callable, str] = "f1",
                   test: Callable = mannwhitneyu, alpha: int = 0.05, cl: float = 0.05,
                   experiment_name: str = "default", db_name: str = "templates",
                   dataset_id: int = None, dataset_name: str = None,
                   output_dir: str = "./output",
                   verbose: bool = False, logger: Logger = None) -> pd.DataFrame:
    """
    Compare machine learning models' performance on the provided data set, using
    cross-validation and statistical hypothesis tests.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression, RidgeClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> import pandas as pd
    >>>
    >>> x, y = make_classification(random_state=42)
    >>>
    >>> models = [LogisticRegression(), RandomForestClassifier(), RidgeClassifier()]
    >>> model_names = ["LogisticRegression", "RandomForestClassifier", "RidgeClassifier"]
    >>> params = [{}, {}, {}]
    >>>
    >>> results = compare_models(models=models, x_train=x, y_train=y, params=params,
    ...                          dataset_id=1, dataset_name="make-class")
    >>>
    >>> pd.set_option('display.width', 9)
    >>> results[['model_name', 'module', 'version', 'ci-l-bound', 'ci-u-bound', 'pvalue']] #doctest: +ELLIPSIS
                   model_name   module version  ci-l-bound ci-u-bound    pvalue
    0      LogisticRegression  sklearn  0.21.2    0.909641          1  1.000000
    1  RandomForestClassifier  sklearn  0.21.2    0.851488          1  0.532541
    2         RidgeClassifier  sklearn  0.21.2    0.896696          1  0.654039

    Parameters
    --------
    :param models: list of machine learning models (keras or sklearn)
    :param x_train: training data
    :param y_train: input labels
    :param params: list of dictionaries, each one containing the arguments of the fit method of a model
    :param x_val: validation data
    :param y_val: validation labels
    :param random_data: if True it enables data randomization
    :param random_model: if True it enables model randomization (if applicable)
    :param seed: seed used to make results reproducible
    :param n_splits: number of cross-validation iterations
    :param scoring: scoring function used to evaluate the model performance (Callable, f1 or accuracy)
    :param test: statistical test
    :param alpha: significance level
    :param cl: confidence level
    :param experiment_name: name of the current experiment
    :param db_name: name of the database
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param output_dir: path to the folder where the results will be saved (as csv file)
    :param verbose: if True enables plots
    :param logger: object used to save progress
    :return: Pandas DataFrame containing a summary of the results
    """

    if not dataset_id:
        dataset_id = 1
        db_name = "new-db"
        drop_db(db_name)

    if not dataset_name:
        dataset_name = "dataset"

    train_cv = []
    val_cv = []

    # Cross-validation
    i = 0
    for model, fit_params in zip(models, params):

        score, fitted_models = cross_validation(model, x=x_train, y=y_train,
                                                x_val=x_val, y_val=y_val,
                                                random_data=random_data, random_model=random_model,
                                                seed=seed, n_splits=n_splits, scoring=scoring, logger=logger,
                                                db_name=db_name, fit_params=fit_params,
                                                dataset_id=dataset_id, dataset_name=dataset_name)

        models[i] = fitted_models[-1]
        train_cv.append(score["train_cv"])
        val_cv.append(score["val_cv"])
        i += 1

    # Find best solution
    best_sol, best_solutions, pvalues = find_best_solution(val_cv,
                                                           test=mannwhitneyu, alpha=alpha,
                                                           use_continuity=False, alternative="two-sided")

    # Compute results' summary
    results = _compute_result_summary(models, random_data, random_model,
                                      seed, n_splits, scoring, test, alpha, cl,
                                      dataset_id, dataset_name,
                                      train_cv, val_cv, pvalues, best_solutions)

    # Save results to csv
    experiment = experiment_name + ".csv"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    results.to_csv(os.path.join(output_dir, experiment))

    # # Plot results
    # if verbose:
    #     _plot_results(val_cv, best_solutions, pvalues, cl)

    return results
