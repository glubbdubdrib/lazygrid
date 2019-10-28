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

import os
import sys
import sqlite3
from abc import ABCMeta
from typing import Union

import joblib
import pickle
import traceback
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from tensorflow import Tensor
from keras import optimizers
import keras


def _get_parameters(model: Union[Sequential, ABCMeta, Pipeline], fit_params: dict) -> tuple:
    """
    Return tuple of strings containing attribute names and model parameters.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> 
    >>> fit_params = {}
    >>> model = RandomForestClassifier()
    >>>
    >>> _get_parameters(model, fit_params=fit_params)
    ('base_estimator', 'DecisionTreeClassifier', 'bootstrap', True, 'criterion', 'gini', 'estimator_params', "('criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'random_state')", 'max_features', 'auto', 'min_samples_leaf', 1, 'min_samples_split', 2, 'n_estimators', 'warn')

    Parameters
    --------
    :param model: machine learning model
    :param fit_params: parameters of the fit method (for keras classifiers)
    :return: model attributes and parameters as a tuple of strings
    """

    parameters = []
    for attribute in dir(model):
        try:
            if not attribute.startswith("_") and not attribute.endswith("_") and \
                    not attribute in ["signature", "history", "trainable_weights", "weights", "output_names", "input_names", "name"]:

                handler = getattr(model, attribute)
                attribute_name = None

                if isinstance(handler, Tensor):
                    continue

                if not callable(handler):

                    if attribute in ["estimator", "base_estimator"]:
                        attribute_name = type(handler).__name__

                    elif " at " in str(handler):
                        attribute_name = str(handler).split(" at ")[0]

                    elif isinstance(handler, list) or isinstance(handler, tuple):

                        if "Tensor" not in str(handler):
                            attribute_name = str(handler)

                    else:
                        attribute_name = handler

                elif callable(handler) and attribute in ["score_func"]:
                    attribute_name = handler.__name__

                if attribute_name:
                    parameters.append(attribute)
                    parameters.append(attribute_name)

        except AttributeError:
            continue

    if isinstance(model, Sequential):

        for key, value in fit_params.items():
            parameters.append(str(key))
            parameters.append(str(value))

        layers = []
        trainable_layers = []
        for layer in model.layers:
            layers.append(layer.output_shape)
            if layer.trainable:
                trainable_layers.append(layer.output_shape)

        parameters.append("layer_shapes")
        parameters.append(layers)

        parameters.append("trainable_layer_shapes")
        parameters.append(trainable_layers)

    return tuple(parameters)


def _get_model_name(model: Union[Sequential, ABCMeta, Pipeline]) -> str:
    """
    Get name of machine learning model.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> model = RandomForestClassifier()
    >>> _get_model_name(model)
    'RandomForestClassifier'

    Parameters
    --------
    :param model: machine learning model
    :return: model name as string
    """

    if isinstance(model, Sequential):
        model_name = str(type(model).__name__)
    else:
        model_name = str(model).split("(")[0]

    return model_name


def _save(model: Union[Sequential, ABCMeta, Pipeline],
          cv_split: int, dataset_id: int, dataset_name: str, fit_params: dict,
          db_name: str = "lazygrid", previous_step_id: int = -1) -> [int, str]:
    """
    Save fitted model into a database.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> x, y = make_classification()
    >>>
    >>> fs = SelectKBest(f_classif, k=5)
    >>> clf = RandomForestClassifier()
    >>> model = Pipeline([('feature_selector', fs), ('clf', clf)])
    >>> type(model.fit(x, y))
    <class 'sklearn.pipeline.Pipeline'>
    >>>
    >>> cv_split = 0
    >>> dataset_id = 1
    >>> dataset_name = "iris"
    >>> fit_params = {}
    >>> db_name = llazygrid   >>> previous_step_id = -1
    >>>
    >>> step_id, model_name = _save(model, cv_split, dataset_id, dataset_name, fit_params, db_name, previous_step_id)
    >>> type(step_id)
    <class 'int'>
    >>> model_name
    'RandomForestClassifier'

    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param fit_params: parameters of the fit method (for keras classifiers)
    :param db_name: database name
    :param previous_step_id: identifier of the previous step (for pipeline classifiers)
    :return: model identifier (int) and model name (str)
    """

    model_name = _get_model_name(model)

    # ---------------------- Create database if does not exists ----------------------

    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, db_name + ".sqlite")
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    stmt = '''CREATE TABLE IF NOT EXISTS MODEL(
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        parameters TEXT NOT NULL,
        dataset_id INTEGER NOT NULL,
        dataset_name TEXT,
        cv_split INTEGER NOT NULL,
        previous_step_id INTEGER,
        fitted_model BLOB NOT NULL,
        UNIQUE (name, parameters, dataset_id, cv_split, previous_step_id)
    )'''
    db.execute(stmt)

    # ---------------------- Create entry document ----------------------

    parameters = _get_parameters(model, fit_params)

    if isinstance(model, Sequential):
        temp = os.path.join("./database", "temp.h5")
        model.save(temp)
        with open(temp, 'rb') as input_file:
            fitted_model = input_file.read()
    else:
        fitted_model = pickle.dumps(model, protocol=2)

    entry = (
        model_name,
        str(parameters),
        dataset_id,
        dataset_name,
        cv_split,
        previous_step_id,
        fitted_model,
    )

    stmt = '''INSERT INTO MODEL(
        name, parameters, dataset_id, dataset_name, cv_split, previous_step_id, fitted_model)
        VALUES(?, ?, ?, ?, ?, ?, ?)'''

    # ---------------------- Insert item into collection ----------------------

    try:

        cursor.execute(stmt, entry)
        step_id = cursor.lastrowid

    except sqlite3.IntegrityError:

        # print(traceback.format_exc())

        query = (
            model_name,
            str(parameters),
            dataset_id,
            cv_split,
            previous_step_id,
        )

        stmt = '''SELECT id FROM MODEL
                  WHERE name=? AND parameters=? AND dataset_id=? AND cv_split=? AND previous_step_id=?'''

        step_id = cursor.execute(stmt, query).fetchone()[0]

    db.commit()
    db.close()

    return step_id, model_name


def save_model(model: Union[Sequential, ABCMeta, Pipeline],
               cv_split: int, dataset_id: int, dataset_name: str,
               fit_params: dict, db_name: str = "templates") -> int:
    """
    Save fitted model into a database. If the model is a sklearn Pipeline, then each step is saved separately.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.datasets import make_classification
    >>>
    >>> x, y = make_classification()
    >>>
    >>> fs = SelectKBest(f_classif, k=5)
    >>> clf = RandomForestClassifier()
    >>> model = Pipeline([('feature_selector', fs), ('clf', clf)])
    >>> type(model.fit(x, y))
    <class 'sklearn.pipeline.Pipeline'>
    >>>
    >>> cv_split = 0
    >>> dataset_id = 1
    >>> dataset_name = "iris"
    >>> fit_params = {}
    >>> db_name = "templates"
    >>>
    >>> signature = save_model(model, cv_split, dataset_id, dataset_name, fit_params, db_name)
    >>> type(signature)
    <class 'int'>

    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param fit_params: parameters of the fit method (for keras classifiers)
    :param db_name: database name
    :return: model signature (identifier of the model)
    """

    if isinstance(model, Pipeline):
        previous_step_id = -1
        previous_step_name = ""
        for step in model.steps:
            previous_step_id, previous_step_name = _save(step[1], cv_split, dataset_id, dataset_name, fit_params,
                                                         db_name, previous_step_id)

    elif isinstance(model, Sequential):
        previous_step_id, previous_step_name = _save(model, cv_split, dataset_id, dataset_name, fit_params, db_name)

    elif model._estimator_type == "classifier" and not isinstance(model, Pipeline):
        previous_step_id, previous_step_name = _save(model, cv_split, dataset_id, dataset_name, fit_params, db_name)

    model.signature = previous_step_id

    return model.signature


def _load(model: Union[Sequential, ABCMeta, Pipeline],
          cv_split: int, dataset_id: int, dataset_name: str, fit_params: dict,
          db_name: str = "templates", previous_step_id: int = -1) -> tuple([Union[Sequential, ABCMeta, Pipeline], int]):
    """
    Load fitted model from a database.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>>
    >>> x, y = make_classification()
    >>>
    >>> fs = SelectKBest(f_classif, k=5)
    >>> clf = RandomForestClassifier()
    >>> model = Pipeline([('feature_selector', fs), ('clf', clf)])
    >>> type(model.fit(x, y))
    <class 'sklearn.pipeline.Pipeline'>
    >>>
    >>> cv_split = 0
    >>> dataset_id = 1
    >>> dataset_name = "iris"
    >>> fit_params = {}
    >>> db_name = "templates"
    >>> previous_step_id = -1
    >>>
    >>> step_id, model_name = _save(model, cv_split, dataset_id, dataset_name, fit_params, db_name, previous_step_id)
    >>>
    >>> model2 = RandomForestClassifier()
    >>> model, step_id = _load(model, cv_split, dataset_id, dataset_name, fit_params, db_name, previous_step_id)

    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param fit_params: parameters of the fit method (for keras classifiers)
    :param db_name: database name
    :param previous_step_id: identifier of the previous step (for pipeline classifiers)
    :return: fitted model and model identifier (int)
    """

    model_name = _get_model_name(model)

    # ---------------------- Connect to database if it exists ----------------------

    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, db_name + ".sqlite")
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    stmt = '''SELECT name FROM sqlite_master WHERE type='table' AND name=? '''
    table = cursor.execute(stmt, ("MODEL",)).fetchone()

    if not table:
        db.close()
        return None, None

    # ---------------------- Connect to database if it exists ----------------------

    parameters = _get_parameters(model, fit_params)

    query = (
        model_name,
        str(parameters),
        dataset_id,
        cv_split,
        previous_step_id,
    )

    stmt = '''SELECT id, fitted_model FROM MODEL
              WHERE name=? AND parameters=? AND dataset_id=? AND cv_split=? AND previous_step_id=?'''

    result = cursor.execute(stmt, query).fetchone()

    # ---------------------- Load fitted model ----------------------

    if not result:
        cursor.close()
        return None, None

    step_id, fitted_model = result

    if isinstance(model, Sequential):
        temp = os.path.join("./database", "temp.h5")
        with open(temp, 'wb') as output_file:
            output_file.write(fitted_model)
        model = keras.models.load_model(temp)
    else:
        model = pickle.loads(fitted_model)
    model.signature = step_id

    db.close()

    return model, step_id


def load_model(model: Union[Sequential, ABCMeta, Pipeline],
               cv_split: int, dataset_id: int, dataset_name: str,
               fit_params: dict, db_name: str = "templates") -> Union[Sequential, ABCMeta, Pipeline]:
    """
    Load fitted model from a database. If the model is a sklearn Pipeline, then each step is loaded separately.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.datasets import make_classification
    >>>
    >>> x, y = make_classification()
    >>>
    >>> fs = SelectKBest(f_classif, k=5)
    >>> clf = RandomForestClassifier()
    >>> model = Pipeline([('feature_selector', fs), ('clf', clf)])
    >>> type(model.fit(x, y))
    <class 'sklearn.pipeline.Pipeline'>
    >>>
    >>> cv_split = 0
    >>> dataset_id = 1
    >>> dataset_name = "iris"
    >>> fit_params = {}
    >>> db_name = "templates"
    >>>
    >>> signature = save_model(model, cv_split, dataset_id, dataset_name, fit_params, db_name)
    >>> model2 = Pipeline([('fs', fs), ('clf', clf)])
    >>> type(load_model(model2, cv_split, dataset_id, dataset_name, fit_params, db_name))
    <class 'sklearn.pipeline.Pipeline'>

    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param fit_params: parameters of the fit method (for keras classifiers)
    :param db_name: database name
    :return: model signature (identifier of the model)
    """

    if isinstance(model, Pipeline):

        step_idx = 0
        previous_step_id = -1
        for step in model.steps:

            fitted_step, previous_step_id = _load(step[1], cv_split, dataset_id,
                                                  dataset_name, fit_params, db_name, previous_step_id)

            if fitted_step:
                model.steps[step_idx] = (model.steps[step_idx][0], fitted_step)
                model.signature = fitted_step.signature

                fitted_model = model
                step_idx += 1

            else:
                fitted_model = None
                break

    elif isinstance(model, Sequential):
        fitted_model, _ = _load(model, cv_split, dataset_id, dataset_name, fit_params, db_name)

    elif model._estimator_type == "classifier" and not isinstance(model, Pipeline):

        fitted_model, _ = _load(model, cv_split, dataset_id, dataset_name, fit_params, db_name)

    if not fitted_model:
        return None

    return fitted_model


def drop_db(db_name: str = "templates") -> None:
    """
    Drop database table if it exists.

    Examples
    --------
    >>> import os
    >>> import sqlite3
    >>>
    >>> drop_db()

    :param db_name: database name
    :return: None
    """

    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, db_name + ".sqlite")

    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    cursor.execute("DROP TABLE IF EXISTS MODEL")

    db.close()

    return

