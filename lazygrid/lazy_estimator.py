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
import os
import pickle
import sys
from typing import Callable, Any, Union, Collection, Iterable
import json
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state, check_memory
from .database import _save_to_db, _load_from_db, load_all_from_db
from .config import create_model_stmt, insert_model_stmt, query_model_stmt


class LazyPipeline(Pipeline):
    """
    A LazyPipeline estimator.

    A lazy pipeline is a sklearn-like pipeline that follows the memoization paradigm.
    Once the pipeline has been fitted, its steps are pickled and stored in a local
    database. Therefore, when the program starts again, the pipeline will fetch its fitted
    steps from the database and will skip the fit operation.

    Parameters
    ----------
    steps
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    database
        Used to cache the fitted transformers of the pipeline.
        It is the path to the database directory.
        Caching the transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    See Also
    --------
    `sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_

    Examples
    --------
    >>> from sklearn import svm
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.feature_selection import f_regression
    >>> from lazygrid.lazy_estimator import LazyPipeline
    >>> # generate some data to play with
    >>> X, y = make_classification(
    ...     n_informative=5, n_redundant=0, random_state=42)
    >>> # ANOVA SVM-C
    >>> anova_filter = SelectKBest(f_regression, k=5)
    >>> clf = svm.SVC(kernel='linear')
    >>> anova_svm = LazyPipeline([('anova', anova_filter), ('svc', clf)])
    >>> # You can set the parameters using the names issued
    >>> # For instance, fit using a k of 10 in the SelectKBest
    >>> # and a parameter 'C' of the svm
    >>> anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
    Pipeline(steps=[('anova', SelectKBest(...)), ('svc', SVC(...))])
    >>> prediction = anova_svm.predict(X)
    >>> anova_svm.score(X, y)
    0.83
    >>> # getting the selected features chosen by anova_filter
    >>> anova_svm['anova'].get_support()
    array([False, False,  True,  True, False, False,  True,  True, False,
           True, False,  True,  True, False,  True, False,  True,  True,
           False, False])
    >>> # Another way to get selected features chosen by anova_filter
    >>> anova_svm.named_steps.anova.get_support()
    array([False, False,  True,  True, False, False,  True,  True, False,
           True, False,  True,  True, False,  True, False,  True,  True,
           False, False])
    >>> # Indexing can also be used to extract a sub-pipeline.
    >>> sub_pipeline = anova_svm[:1]
    >>> sub_pipeline
    Pipeline(steps=[('anova', SelectKBest(...))])
    >>> coef = anova_svm[-1].coef_
    >>> anova_svm['svc'] is anova_svm[-1]
    True
    >>> coef.shape
    (1, 10)
    >>> sub_pipeline.inverse_transform(coef).shape
    (1, 20)
    """

    def __init__(self, steps, database: str = "./database/", verbose: bool = False):
        super().__init__(steps, memory=None, verbose=verbose)
        self.database = database

    # TODO: fit_params sould be checked
    def fit(self, X: Iterable, y: Iterable = None, data_set: int = 0, train: Iterable = None, **fit_params):
        """Fit the model

        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        data_set
            Data set identifier. It must be different from data set to data set.

        train
            Training set indexes.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self
            This estimator
        """

        try:
            check_is_fitted(self, "is_fitted_")

        except sklearn.exceptions.NotFittedError:

            # Check that X and y have correct shape
            X, y = check_X_y(X, y)
            self._validate_steps()

            if train is None:
                try:
                    # get train indeces from caller
                    callingframe = sys._getframe(1)
                    train = callingframe.f_locals["train"]

                except KeyError:
                    train = np.arange(0, len(X))

            self.train_ = train
            self.data_set_ = data_set
            self.database_ = os.path.join(self.database, "database.sqlite")
            self.fit_params_ = fit_params
            self._load()
            self._fit(X, y, **fit_params)
            self._save()
            self.is_fitted_ = True

        return self

    def _fit(self, X, y=None, **fit_params):
        Xt = X
        for (step_idx, name, transformer) in self._iter(with_final=False, filter_passthrough=False):
            if not hasattr(transformer, "is_fitted_"):
                Xt = transformer.fit_transform(Xt, y, **fit_params)
                self.steps[step_idx] = (name, transformer)

        if not hasattr(self.steps[-1][1], "is_fitted_"):
            self.steps[-1][1].fit(Xt, y, **fit_params)

        return self

    def _save(self):

        parameters = {}
        step_ids = []
        previous_id = None
        for step in self.steps:
            estimator = step[1]
            estimator.is_fitted_ = True
            pms = estimator.get_params()
            for key, value in pms.items():
                if isinstance(value, Callable):
                    pms[key] = value.__name__

                if value == "warn":
                    pms[key] = 10

            step_name = estimator.__class__.__name__
            parameters[step_name] = pms
            query = (
                self.data_set_,
                json.dumps(self.train_.tolist()),
                json.dumps(pms),
                json.dumps([previous_id]),
            )
            entry = (
                *query,
                pickle.dumps(estimator),
            )
            result = _save_to_db(self.database_, entry, query, create_model_stmt, insert_model_stmt, query_model_stmt)
            if result:
                previous_id = result[0]
                step_ids.append(previous_id)

        self.parameters_ = json.dumps(parameters)
        self.model_ids_ = step_ids

        check = load_all_from_db(self.database_)

        return self

    def _load(self):

        parameters = {}
        step_ids = []
        previous_id = None
        i = 0
        for step in self.steps:
            estimator = step[1]
            pms = estimator.get_params()
            for key, value in pms.items():
                if isinstance(value, Callable):
                    pms[key] = value.__name__

                if value == "warn":
                    pms[key] = 10

            step_name = estimator.__class__.__name__
            parameters[step_name] = pms
            query = (
                self.data_set_,
                json.dumps(self.train_.tolist()),
                json.dumps(pms),
                json.dumps([previous_id]),
            )
            result = _load_from_db(self.database_, query, create_model_stmt, query_model_stmt)
            if result:
                previous_id = result[0]
                step_ids.append(previous_id)
                estimator = pickle.loads(result[5])
                estimator.is_fitted_ = True
                self.steps[i] = (self.steps[i][0], copy.deepcopy(estimator))

            else:
                break

            i += 1

        self.parameters_ = json.dumps(parameters)
        self.model_ids_ = step_ids

        check = load_all_from_db(self.database_)

        return self
