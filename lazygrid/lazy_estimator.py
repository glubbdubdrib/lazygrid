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
import json
import os
import pickle
from collections import OrderedDict
from typing import Callable, Iterable, List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y

from .config import create_model_stmt, insert_model_stmt, query_model_stmt
from .database import _save_to_db, _load_from_db


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
    >>> import pandas as pd
    >>> # generate some data to play with
    >>> X, y = make_classification(
    ...     n_informative=5, n_redundant=0, random_state=42)
    >>> X = pd.DataFrame(X)
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
    def fit(self, X: pd.DataFrame, y: Iterable = None, **fit_params):
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

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self
            This estimator
        """

        assert isinstance(X, pd.DataFrame)

        # Check that X and y have correct shape
        Xnp, ynp = check_X_y(X, y)
        self._validate_steps()

        self.train_ = tuple(X.index)
        self.features_ = tuple(X.columns)
        self.database_ = os.path.join(self.database, "database.sqlite")
        self.fit_params_ = fit_params
        self._fit(X, y, **fit_params)
        self.is_fitted_ = True

        return self

    def _fit(self, X: pd.DataFrame, y: Iterable = None, **fit_params):
        Xt = X
        ids = ()
        # fit or load intermediate steps
        for (step_idx, name, transformer) in self._iter(with_final=False, filter_passthrough=False):
            transformer, ids, Xt = self._fit_step(transformer, ids, False, Xt, y, **fit_params)
            self.steps[step_idx] = (name, copy.deepcopy(transformer))

        # fit or load final step
        transformer, ids, Xt = self._fit_step(self.steps[-1][1], ids, True, Xt, y, **fit_params)
        self.steps[-1] = (self.steps[-1][0], copy.deepcopy(transformer))

        return self

    def _fit_step(self, transformer: BaseEstimator, ids: Tuple, is_final: bool,
                  X: pd.DataFrame, y: Iterable = None, **fit_params):
        # make transformer unique for each CV split
        transformer.train_ = tuple(X.index)
        transformer.features_ = tuple(X.columns)

        # load transformer from database
        transformer_loaded, ids_loaded = self._load(transformer, ids)
        is_loaded = False if transformer_loaded is None else True
        if is_loaded:
            transformer = transformer_loaded
            ids = ids_loaded

        # fit final step
        if is_final:
            if not is_loaded:
                transformer.fit(X, y, **fit_params)

        # fit intermediate steps
        else:
            if not is_loaded:
                transformer.fit(X, y, **fit_params)

            transformed_data = transformer.transform(X)

            if isinstance(transformed_data, Tuple):
                X, y = transformed_data

            else:
                Xnp = transformed_data

                # reshape input data
                if Xnp.shape != X.shape:
                    if isinstance(X, pd.DataFrame):
                        X = X.iloc[:, transformer.get_support()]

                else:
                    X = pd.DataFrame(Xnp)

        # save transformer
        if not is_loaded:
            ids = self._save(transformer, ids)

        return transformer, ids, X

    def _save(self, transformer: BaseEstimator, ids: Tuple):
        query, entry = _step_db(transformer, ids)
        result = _save_to_db(self.database_, entry, query, create_model_stmt, insert_model_stmt, query_model_stmt)
        if result:
            ids = ids + (result[0],)
        return ids

    def _load(self, transformer: BaseEstimator, ids: Tuple):
        query, entry = _step_db(transformer, ids)
        result = _load_from_db(self.database_, query, create_model_stmt, query_model_stmt)
        if result:
            ids = ids + (result[0],)
            transformer = pickle.loads(result[5])
            transformer.is_fitted_ = True
            return transformer, ids
        else:
            return None, None


def _step_db(estimator: BaseEstimator, ids: Tuple):
    estimator.is_fitted_ = True
    # make a dictionary of parameters
    pms = ()
    estimator_params = estimator.get_params()
    for key in sorted(estimator_params.keys()):
        value = estimator_params[key]
        if isinstance(value, Callable):
            pms = pms + (key, value.__name__)

        if value == "warn":
            pms = pms + (key, 10)

        # discard parameters which are not json serializable
        try:
            json.dumps(value)
            pms = pms + (key, value)
        except TypeError:
            continue

    query = (
        json.dumps(estimator.train_),
        json.dumps(estimator.features_),
        json.dumps(pms),
        json.dumps(ids),
    )
    entry = (
        *query,
        pickle.dumps(estimator),
    )

    return query, entry
