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
from typing import Callable, Any, Union, Collection
import json
import keras
import numpy as np
import sklearn
from keras import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import tensorflow as tf
from sklearn.pipeline import Pipeline
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state, check_memory
from .database import drop_db, _save_to_db, _load_from_db, load_all_from_db
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
    steps : List
        The estimator to become lazy.
    database : str
        Used to cache the fitted transformers of the pipeline.
    verbose : int
        The random state of the estimator.

    Attributes
    ----------


    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomForestClassifier(max_depth=2, random_state=0)
    >>> clf.fit(X, y)

    Notes
    -----

    See Also
    --------

    """

    def __init__(self, steps, database: str = "./database/", verbose: bool = False):
        super().__init__(steps, memory=None, verbose=verbose)
        self.database = database

    # TODO: fit_params sould be checked
    def fit(self, X, y=None, data_set=0, train=None, **fit_params):
        """
        Fit model with some samples.

        Parameters
        --------
        :param x: train data
        :param y: train labels
        :return: None
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
