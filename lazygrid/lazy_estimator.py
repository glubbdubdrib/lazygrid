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
            if transformer is None or transformer == 'passthrough':
                with _print_elapsed_time('Pipeline', self._log_message(step_idx)):
                    continue

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





    # def predict(self, x) -> Any:
    #     """
    #     Predict labels for some input samples.
    #
    #     Parameters
    #     --------
    #     :param x: input data
    #     :return: predictions
    #     """
    #
    #     # Check is fit had been called
    #     check_is_fitted(self, "is_fitted_")
    #     # Input validation
    #     x = check_array(x)
    #
    #     return self.estimator.predict(x)
    #
    # def score(self, X, y, sample_weight=None):
    #     """
    #     Compute score for some input samples.
    #
    #     Parameters
    #     --------
    #     :param x: input data
    #     :param y: input labels
    #     :return: score
    #     """
    #
    #     # Check is fit had been called
    #     check_is_fitted(self, "is_fitted_")
    #     # Check that X and y have correct shape
    #     X, y = check_X_y(X, y)
    #
    #     return self.estimator.score(X, y, sample_weight)



#     def _get_entry(self) -> tuple:
#         """
#         Define database entry for the model.
#
#         Parameters
#         --------
#         :return: database entry
#         """
#
#         entry = (
#             self.model_name,
#             self.model_type,
#             pickle.dumps(self.__class__, protocol=2),
#             str(self.parameters),
#             self.fit_parameters,
#             self.predict_parameters,
#             self.score_parameters,
#             self.version,
#             int(self.is_standalone),
#             str(self.models_id),
#             self.dataset_id,
#             self.dataset_name,
#             self.cv_split,
#             self.previous_step_id,
#             self.serialized_model,
#         )
#         return entry
#
#     def _get_query(self) -> tuple:
#         """
#         Define database query for the model.
#
#         Parameters
#         --------
#         :return: database query
#         """
#         query = (
#             self.model_name,
#             self.model_type,
#             str(self.parameters),
#             self.fit_parameters,
#             self.predict_parameters,
#             self.score_parameters,
#             self.version,
#             str(self.models_id),
#             self.dataset_id,
#             self.cv_split,
#             self.previous_step_id,
#         )
#         return query
#
#     def _to_database(self):
#         """
#         Send model into a database.
#
#         Parameters
#         --------
#         :return: query result
#         """
#         self.entry = self._get_entry()
#         self.query = self._get_query()
#         return save_to_db(self.db_name, self.entry, self.query, create_model_stmt, insert_model_stmt, query_model_stmt)
#
#     def _from_database(self, **kwargs):
#         """
#         Fetch model from a database.
#
#         Parameters
#         --------
#         :return: query result
#         """
#         self.query = self._get_query()
#         return load_from_db(self.db_name, self.query, create_model_stmt, query_model_stmt)
#
#     def _load_model(self) -> Any:
#         """
#         Load model from database if possible.
#
#         Parameters
#         --------
#         :return: wrapped model
#         """
#         result = self._from_database()
#         if result:
#             model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
#             self.is_fitted = True
#             self.model_id = model_id
#             self.model = pickle.loads(serialized_model)
#             self.serialized_model = serialized_model
#
#         return self
#
#     def _save_model(self) -> None:
#         """
#         Save model into database.
#
#         Parameters
#         --------
#         :return: None
#         """
#         if not self.serialized_model:
#             self.serialized_model = pickle.dumps(self.model, protocol=2)
#
#         result = self._to_database()
#         if result:
#             model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
#             self.model_id = model_id
#
#
#
# class SklearnWrapper(Wrapper):
#     """
#     Class to wrap sklearn models.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>> from sklearn.linear_model import LogisticRegression
#     >>>
#     >>> lg_model = lg.wrapper.SklearnWrapper(LogisticRegression())
#     """
#
#     def __init__(self, model: ClassifierMixin, dataset_id: int = 0, dataset_name: str = "default-data-set",
#                  db_name: str = "./database/default-db.sqlite",
#                  fit_params: dict = None, predict_params: dict = None, score_params: dict = None,
#                  model_id: int = None, cv_split: int = None, is_standalone: bool = True):
#         """
#         Wrapper initialization.
#
#         Parameters
#         --------
#         :param model: machine learning model
#         :param dataset_id: data set identifier
#         :param dataset_name: data set name
#         :param db_name: database name
#         :param fit_params: model's fit parameters
#         :param predict_params: model's predict parameters
#         :param score_params: model's score parameters
#         :param model_id: model identifier
#         :param cv_split: cross-validation split identifier
#         :param is_standalone: True if model can be used independently from other models
#         """
#         model = corner_cases(model)
#         Wrapper.__init__(self, model, dataset_id, dataset_name, db_name, fit_params,
#                          predict_params, score_params, model_id, cv_split, is_standalone)
#         self.parameters = self.parse_parameters()
#
#     def set_random_seed(self, seed, split_index, random_model):
#         """
#         Set model random state if possible.
#
#         Parameters
#         --------
#         :param seed: random seed
#         :param split_index: cross-validation split identifier
#         :param random_model: whether the model should have the same random state for each cross-validation split
#         :return: None
#         """
#         if random_model:
#             random_state = split_index
#         else:
#             random_state = seed
#         try:
#             self.model.set_params(**{"random_state": random_state})
#         except (AttributeError, ValueError):
#             pass
#         self.cv_split = split_index
#         self.parameters = self.parse_parameters()
#
#     def parse_parameters(self) -> str:
#         """
#         Parse sklearn model parameters.
#
#         Parameters
#         --------
#         :return: model parameters
#         """
#         return parse_sklearn_model(self.model)
#
#
# class PipelineWrapper(Wrapper):
#     """
#     Class to wrap sklearn pipeline objects.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>> from sklearn.ensemble import RandomForestClassifier
#     >>> from sklearn.pipeline import Pipeline
#     >>> from sklearn.preprocessing import StandardScaler
#     >>> from sklearn.feature_selection import SelectKBest, mutual_info_classif
#     >>>
#     >>> standardizer = StandardScaler()
#     >>> feature_selector = SelectKBest(score_func=mutual_info_classif, k=2)
#     >>> classifier = RandomForestClassifier(random_state=42)
#     >>> pipeline = Pipeline(steps=[("standardizer", standardizer),
#     ...                            ("feature_selector", feature_selector),
#     ...                            ("classifier", classifier)])
#     >>>
#     >>> lg_model = lg.wrapper.PipelineWrapper(pipeline)
#     """
#
#     def __init__(self, model: Pipeline, dataset_id: int = 0, dataset_name: str = "default-data-set",
#                  db_name: str = "./database/default-db.sqlite",
#                  fit_params: dict = None, predict_params: dict = None, score_params: dict = None,
#                  model_id: int = None, cv_split: int = None, is_standalone: bool = True):
#         """
#         Wrapper initialization.
#
#         Parameters
#         --------
#         :param model: machine learning model
#         :param dataset_id: data set identifier
#         :param dataset_name: data set name
#         :param db_name: database name
#         :param fit_params: model's fit parameters
#         :param predict_params: model's predict parameters
#         :param score_params: model's score parameters
#         :param model_id: model identifier
#         :param cv_split: cross-validation split identifier
#         :param is_standalone: True if model can be used independently from other models
#         """
#         model = corner_cases(model)
#         Wrapper.__init__(self, model, dataset_id, dataset_name, db_name, fit_params,
#                          predict_params, score_params, model_id, cv_split, is_standalone)
#         self.models = []
#         self.models_id = []
#         for step in model.steps:
#             pipeline_step = SklearnWrapper(model=step[1], cv_split=cv_split, dataset_id=dataset_id,
#                                            dataset_name=dataset_name, db_name=db_name, is_standalone=False)
#             self.models.append(pipeline_step)
#             self.models_id.append(pipeline_step.model_id)
#
#         parameters = []
#         for step in self.models:
#             parameters.append(step.parameters)
#         self.parameters = str(parameters)
#
#     def save_model(self) -> None:
#         """
#         Save model into database.
#
#         Parameters
#         --------
#         :return: None
#         """
#         self.models_id = []
#         previous_step_id = -1
#         # save each step separately
#         for model in self.models:
#             model.previous_step_id = previous_step_id
#             model.save_model()
#             self.models_id.append(model.model_id)
#             previous_step_id = model.model_id
#         # serialize model
#         if not self.serialized_model:
#             self.serialized_model = pickle.dumps(self.model, protocol=2)
#         result = self.to_database()
#         if result:
#             model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
#             self.model_id = model_id
#
#     def load_model(self) -> Any:
#         """
#         Load model from database.
#
#         Parameters
#         --------
#         :return: None
#         """
#         pipeline = []
#         previous_step_id = -1
#         i = 0
#         is_fetchable = True
#         # load each step separately (if present)
#         for step in self.models:
#             step.previous_step_id = previous_step_id
#             if is_fetchable:
#                 fitted_step = step.load_model()
#                 if fitted_step.model_id and is_fetchable:
#                     pipeline_step = ("id_" + str(fitted_step.model_id), fitted_step.model)
#                     previous_step_id = fitted_step.model_id
#                     self.models[i].is_fitted = True
#                     self.models_id[i] = previous_step_id
#                 else:
#                     # from now on the pipeline will be different, you can't fetch from database other models
#                     is_fetchable = False
#             if not is_fetchable:
#                 pipeline_step = ("n_" + str(i), copy.deepcopy(step.model))
#             pipeline.append(pipeline_step)
#             i += 1
#         self.model.steps = pipeline
#         result = self.from_database()
#         if result:
#             model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
#             self.is_fitted = True
#             self.model_id = model_id
#             self.model = pickle.loads(serialized_model)
#             self.serialized_model = serialized_model
#         return self
#
#     def set_random_seed(self, seed, split_index, random_model):
#         """
#         Set model random state if possible.
#
#         Parameters
#         --------
#         :param seed: random seed
#         :param split_index: cross-validation split identifier
#         :param random_model: whether the model should have the same random state for each cross-validation split
#         :return: None
#         """
#         if random_model:
#             random_state = split_index
#         else:
#             random_state = seed
#         for parameter in list(self.model.get_params().keys()):
#             if "random_state" in parameter:
#                 self.model.set_params(**{parameter: random_state})
#         self.cv_split = split_index
#         # set random seed of pipeline steps
#         for model in self.models:
#             model.set_random_seed(seed, split_index, random_model)
#
#     def parse_parameters(self, **kwargs) -> str:
#         pass
#
#     def fit(self, x_train, y_train, **kwargs):
#         """
#         Fit model with some samples.
#
#         Parameters
#         --------
#         :param x_train: train data
#         :param y_train: train labels
#         :return: None
#         """
#         x_train_t = x_train
#         i = 0
#         for pipeline_step, model in zip(self.model.steps, self.models):
#             if not model.is_fitted:
#                 # print("NOT FITTED!")
#                 pipeline_step[1].fit(x_train_t, y_train, )
#                 self.models[i] = SklearnWrapper(model=copy.deepcopy(pipeline_step[1]), cv_split=self.cv_split,
#                                                 dataset_id=self.dataset_id, dataset_name=self.dataset_name,
#                                                 db_name=self.db_name, is_standalone=False)
#                 self.models[i].is_fitted = True
#             if hasattr(pipeline_step[1], "transform"):
#                 x_train_t = pipeline_step[1].transform(x_train_t)
#             i += 1
#         self.is_fitted = True
#
#
# class KerasWrapper(Wrapper):
#     """
#     Class to wrap keras objects.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>>
#     >>> keras_nn = lg.neural_models.keras_classifier(layers=[10, 5], input_shape=(20,), n_classes=2, verbose=False)
#     >>> lg_model = lg.wrapper.KerasWrapper(keras_nn)
#     """
#
#     def __init__(self, model: Model, dataset_id: int = 0, dataset_name: str = "default-data-set",
#                  db_name: str = "./database/default-db.sqlite",
#                  fit_params: dict = None, predict_params: dict = None, score_params: dict = None,
#                  model_id: int = None, cv_split: int = None, is_standalone: bool = True):
#         """
#         Wrapper initialization.
#
#         Parameters
#         --------
#         :param model: machine learning model
#         :param dataset_id: data set identifier
#         :param dataset_name: data set name
#         :param db_name: database name
#         :param fit_params: model's fit parameters
#         :param predict_params: model's predict parameters
#         :param score_params: model's score parameters
#         :param model_id: model identifier
#         :param cv_split: cross-validation split identifier
#         :param is_standalone: True if model can be used independently from other models
#         """
#         Wrapper.__init__(self, model, dataset_id, dataset_name, db_name, fit_params,
#                          predict_params, score_params, model_id, cv_split, is_standalone)
#         self.parameters = self.parse_parameters()
#
#     def load_model(self) -> Any:
#         """
#         Load model from database.
#
#         Parameters
#         --------
#         :return: None
#         """
#         result = self.from_database()
#         if result:
#             model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
#             self.is_fitted = True
#             self.model_id = model_id
#             self.model = load_neural_model(serialized_model)
#             self.serialized_model = serialized_model
#         else:
#             self.is_fitted = False
#
#     def save_model(self) -> None:
#         """
#         Save model into database.
#
#         Parameters
#         --------
#         :return: None
#         """
#         if not self.serialized_model:
#             self.serialized_model = save_neural_model(self.model)
#         result = self.to_database()
#         if result:
#             model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
#             self.model_id = model_id
#
#     def set_random_seed(self, seed: int, split_index, random_model):
#         """
#         Set model random state if possible.
#
#         Parameters
#         --------
#         :param seed: random seed
#         :param split_index: cross-validation split identifier
#         :param random_model: whether the model should have the same random state for each cross-validation split
#         :return: None
#         """
#         if random_model:
#             random_state = split_index
#         else:
#             random_state = seed
#         tf.set_random_seed(seed=random_state)
#         reset_weights(self.model, random_state)
#         self.cv_split = split_index
#         self.parameters = self.parse_parameters()
#
#     def parse_parameters(self) -> str:
#         """
#         Parse model parameters.
#
#         Parameters
#         --------
#         :return: model parameters
#         """
#         return parse_neural_model(self.model)
#
#     def fit(self, x, y, **kwargs):
#         """
#         Fit model with some samples.
#
#         Parameters
#         --------
#         :param x: train data
#         :param y: train labels
#         :return: None
#         """
#         if not self.is_fitted:
#             fit_params = json.loads(self.fit_parameters)
#             self.model.fit(x, y, **fit_params)
#             self.is_fitted = True
#
#     def predict(self, x, **kwargs) -> Any:
#         """
#         Predict labels for some input samples.
#
#         Parameters
#         --------
#         :param x: input data
#         :return: predictions
#         """
#         predict_params = json.loads(self.predict_parameters)
#         return self.model.predict(x, **predict_params)
#
#     def score(self, x, y, **kwargs) -> Any:
#         """
#         Compute score for some input samples.
#
#         Parameters
#         --------
#         :param x: input data
#         :param y: input labels
#         :return: score
#         """
#         score_params = json.loads(self.score_parameters)
#         return self.model.score(x, y, **score_params)
#
#
# def parse_sklearn_model(model):
#     """
#     Parse sklearn model parameters.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>> from sklearn.linear_model import LogisticRegression
#     >>>
#     >>> lg_model = lg.wrapper.SklearnWrapper(LogisticRegression())
#     >>> parameters = lg.wrapper.parse_sklearn_model(lg_model)
#
#     Parameters
#     --------
#     :param model: sklearn model
#     :return: None
#     """
#     args = []
#     for attribute_name in dir(model):
#         if not attribute_name.startswith("_") and not attribute_name.endswith("_"):
#             try:
#                 attribute = getattr(model, attribute_name)
#                 if isinstance(attribute, Callable):
#                     attribute = attribute.__name__
#                 else:
#                     attribute = ' '.join(str(attribute).split())
#                 if attribute_name != str(attribute):
#                     args.append(attribute_name + ": " + str(attribute))
#             except (sklearn.exceptions.NotFittedError, AttributeError):
#                 pass
#     return ", ".join(args)
#
#
# def parse_neural_model(model: Model) -> str:
#     """
#     Parse keras/tensorflow model parameters.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>>
#     >>> keras_nn = lg.neural_models.keras_classifier(layers=[10, 5], input_shape=(20,), n_classes=2, verbose=False)
#     >>> parameters = lg.wrapper.parse_neural_model(keras_nn)
#
#     Parameters
#     --------
#     :param model: neural network model
#     :return: None
#     """
#     parameters = []
#     parameters.append("input_shape: " + str(model.input_shape))
#     parameters.append("output_shape: " + str(model.output_shape))
#
#     layers = []
#     trainable_layers = []
#     for layer in model.layers:
#         layers.append(layer.output_shape)
#         if layer.trainable:
#             trainable_layers.append(layer.output_shape)
#
#     parameters.append("layers: " + str(layers))
#     parameters.append("trainable_layers: " + str(trainable_layers))
#
#     loss_functions = []
#     for item in model.loss_functions:
#         loss_functions.append(item.__name__)
#     loss_functions.sort()
#     parameters.append("loss_functions: " + str(loss_functions))
#
#     losses = []
#     for item in model.losses:
#         losses.append(item.__name__)
#     losses.sort()
#     parameters.append("losses: " + str(losses))
#
#     metrics = []
#     for item in model.metrics:
#         metrics.append(item)
#     metrics.sort()
#     parameters.append("metrics: " + str(metrics))
#
#     metrics_names = []
#     for item in model.metrics_names:
#         metrics_names.append(item)
#     metrics_names.sort()
#     parameters.append("metrics_names: " + str(metrics_names))
#
#     parameters.append("optimizer_class: " + model.optimizer.__class__.__name__)
#     optimizer_parameters = []
#     for attribute_name in dir(model.optimizer):
#         attribute = getattr(model.optimizer, attribute_name)
#         if type(attribute) is float:
#             optimizer_parameters.append(str(attribute_name) + ": " + str(attribute))
#     parameters.append("optimizer_params: " + str(optimizer_parameters))
#
#     return str(parameters)
#
#
# def corner_cases(model: Any) -> Any:
#     """
#     Check parameter synonyms.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>> from sklearn.ensemble import RandomForestClassifier
#     >>>
#     >>> model_1 = lg.wrapper.corner_cases(RandomForestClassifier())
#     >>> model_2 = lg.wrapper.corner_cases(RandomForestClassifier(n_estimators=10))
#     >>>
#     >>> model_1.n_estimators == model_2.n_estimators
#     True
#
#     Parameters
#     --------
#     :param model: sklearn model
#     :return: model
#     """
#     if isinstance(model, RandomForestClassifier):
#         if getattr(model, "n_estimators") == "warn":
#             setattr(model, "n_estimators", 10)
#     return model
#
#
# def dict_to_json(dictionary: Union[dict, str]) -> str:
#     """
#     Sort dictionary by key and transform it into a string.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>>
#     >>> my_string = lg.wrapper.dict_to_json({"C": 1, "B": 2, "A": 3})
#
#     Parameters
#     --------
#     :param dictionary: python dictionary
#     :return: sorted dictionary as string
#     """
#     if isinstance(dictionary, str):
#         dictionary = json.loads(dictionary)
#     # sort dictionary by key
#     if dictionary:
#         sorted_dictionary = {}
#         for param_key in sorted(dictionary):
#             sorted_dictionary[param_key] = dictionary[param_key]
#     else:
#         sorted_dictionary = {}
#     return json.dumps(sorted_dictionary)
#
#
# def load_neural_model(model_bytes) -> Model:
#     """
#     Load keras model from binaries.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>>
#     >>> keras_nn = lg.neural_models.keras_classifier(layers=[10, 5], input_shape=(20,), n_classes=2, verbose=False)
#     >>>
#     >>> tmp_file = "nn_binary.h5"
#     >>> keras_nn.save(tmp_file)
#     >>> with open(tmp_file, 'rb') as input_file:
#     ...     keras_nn_binary = input_file.read()
#     >>>
#     >>> keras_nn_recovered = lg.wrapper.load_neural_model(keras_nn_binary)
#
#     Parameters
#     --------
#     :param model_bytes: serialized keras model
#     :return: keras model
#     """
#     temp = os.path.join("./tmp", "temp.h5")
#     with open(temp, 'wb') as output_file:
#         output_file.write(model_bytes)
#     return keras.models.load_model(temp)
#
#
# def save_neural_model(model: Model) -> Model:
#     """
#     Serialize keras model.
#
#     Examples
#     --------
#     >>> import lazygrid as lg
#     >>>
#     >>> keras_nn = lg.neural_models.keras_classifier(layers=[10, 5], input_shape=(20,), n_classes=2, verbose=False)
#     >>>
#     >>> keras_nn_binary = lg.wrapper.save_neural_model(keras_nn)
#
#     Parameters
#     --------
#     :param model: keras model
#     :return: serialized keras model
#     """
#     if not os.path.isdir("./tmp"):
#         os.mkdir("./tmp")
#     temp = os.path.join("./tmp", "temp.h5")
#     model.save(temp)
#     with open(temp, 'rb') as input_file:
#         fitted_model = input_file.read()
#     return fitted_model
