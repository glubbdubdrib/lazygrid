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
import sqlite3
import json
import pickle
import traceback
from sklearn.pipeline import Pipeline
import keras
from .wrapper import ModelWrapper


def _save(model: ModelWrapper, cv_split: int, dataset_id: int, dataset_name: str,
          db_name: str = "templates", previous_step_id: int = -1) -> int:
    """
    Save fitted model into a database.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> import lazygrid as lg
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
    >>> db_name = "lazygrid"
    >>> previous_step_id = -1
    >>> model = lg.ModelWrapper(model)
    >>>
    >>> step_id = _save(model, cv_split, dataset_id, dataset_name, db_name, previous_step_id)
    >>> type(step_id)
    <class 'int'>

    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param db_name: database name
    :param previous_step_id: identifier of the previous step (for pipeline classifiers)
    :return: model identifier
    """

    # Create database if does not exists
    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, db_name + ".sqlite")
    db = sqlite3.connect(db_name)
    cursor = db.cursor()
    stmt = '''CREATE TABLE IF NOT EXISTS MODEL(
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        parameters TEXT NOT NULL,
        fit_parameters TEXT NOT NULL,
        version TEXT NOT NULL,
        is_standalone INTEGER NOT NULL,
        models_id TEXT NOT NULL,
        dataset_id INTEGER NOT NULL,
        dataset_name TEXT,
        cv_split INTEGER NOT NULL,
        previous_step_id INTEGER,
        fitted_model BLOB NOT NULL,
        UNIQUE (name, type, parameters, fit_parameters, version, models_id, dataset_id, cv_split, previous_step_id)
    )'''
    db.execute(stmt)

    # Serialize model
    if model.model_type in ["keras", "tensorflow"]:
        temp = os.path.join("./database", "temp.h5")
        model.model.save(temp)
        with open(temp, 'rb') as input_file:
            fitted_model = input_file.read()
    else:
        fitted_model = pickle.dumps(model.model, protocol=2)

    # define entry
    entry = (
        model.model_name,
        model.model_type,
        str(model.parameters),
        model.fit_parameters,
        model.version,
        int(model.is_standalone),
        str(model.models_id),
        dataset_id,
        dataset_name,
        cv_split,
        previous_step_id,
        fitted_model,
    )
    stmt = '''INSERT INTO MODEL(
        name, type, parameters, fit_parameters, version, is_standalone, models_id,
        dataset_id, dataset_name, cv_split, previous_step_id, fitted_model)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

    # Insert item
    try:
        cursor.execute(stmt, entry)
        step_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        # print(traceback.format_exc())
        query = (
            model.model_name,
            model.model_type,
            str(model.parameters),
            model.fit_parameters,
            model.version,
            str(model.models_id),
            dataset_id,
            cv_split,
            previous_step_id,
        )
        stmt = '''SELECT id FROM MODEL
                  WHERE name=? AND type=? AND parameters=? AND fit_parameters=? AND version=? AND models_id=? AND 
                        dataset_id=? AND cv_split=? AND previous_step_id=?'''
        step_id = cursor.execute(stmt, query).fetchone()[0]

    db.commit()
    db.close()

    return step_id


def save_model(model: ModelWrapper,
               cv_split: int, dataset_id: int, dataset_name: str,
               db_name: str = "templates"):
    """
    Save fitted model into a database. If the model is a sklearn Pipeline, then each step is saved separately.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.datasets import make_classification
    >>> import lazygrid as lg
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
    >>> model = lg.ModelWrapper(model)
    >>>
    >>> save_model(model, cv_split, dataset_id, dataset_name, db_name)

    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param db_name: database name
    :return: None
    """

    if model.model_name is "Pipeline":
        previous_step_id = -1
        for step in model.models:
            previous_step_id = _save(step, cv_split, dataset_id, dataset_name, db_name, previous_step_id)

    _save(model, cv_split, dataset_id, dataset_name, db_name)


def _load(model: ModelWrapper,
          cv_split: int, dataset_id: int, dataset_name: str,
          db_name: str = "templates", previous_step_id: int = -1) -> ModelWrapper:
    """
    Load fitted model from a database.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.datasets import make_classification
    >>> import lazygrid as lg
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
    >>> model = lg.ModelWrapper(model)
    >>>
    >>> model_id = _save(model, cv_split, dataset_id, dataset_name, db_name, previous_step_id)
    >>>
    >>> model = _load(model, cv_split, dataset_id, dataset_name, db_name, previous_step_id)
    >>> type(model)
    <class 'wrapper.ModelWrapper'>


    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param db_name: database name
    :param previous_step_id: identifier of the previous step (for pipeline classifiers)
    :return: fitted model
    """

    # Connect to database if it exists
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
        return None

    # Define query
    query = (
        model.model_name,
        model.model_type,
        str(model.parameters),
        model.fit_parameters,
        model.version,
        str(model.models_id),
        dataset_id,
        cv_split,
        previous_step_id,
    )
    stmt = '''SELECT id, fitted_model FROM MODEL
                      WHERE name=? AND type=? AND parameters=? AND fit_parameters=? AND version=? AND models_id=? AND
                            dataset_id=? AND cv_split=? AND previous_step_id=?'''
    result = cursor.execute(stmt, query).fetchone()

    # Load fitted model if present
    if not result:
        cursor.close()
        return None

    model_id, model_bytes = result

    # load fitted model
    if model.model_type in ["keras", "tensorflow"]:
        temp = os.path.join("./database", "temp.h5")
        with open(temp, 'wb') as output_file:
            output_file.write(model_bytes)
        fitted_model = keras.models.load_model(temp)
    else:
        fitted_model = pickle.loads(model_bytes)
    model = ModelWrapper(fitted_model, model.fit_parameters, model_id)

    db.close()

    return model


def load_model(model: ModelWrapper,
               cv_split: int, dataset_id: int, dataset_name: str,
               db_name: str = "templates") -> ModelWrapper:
    """
    Load fitted model from a database. If the model is a sklearn Pipeline, then each step is loaded separately.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.datasets import make_classification
    >>> import lazygrid as lg
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
    >>> model = lg.ModelWrapper(model)
    >>>
    >>> save_model(model, cv_split, dataset_id, dataset_name, db_name)
    >>>
    >>> model = load_model(model, cv_split, dataset_id, dataset_name, db_name)
    >>> type(model)
    <class 'wrapper.ModelWrapper'>

    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param db_name: database name
    :return: model signature (identifier of the model)
    """

    if model.model_name is "Pipeline":

        pipeline = []
        previous_step_id = -1
        i = 0

        # load one step at a time
        for step in model.models:

            fitted_step = _load(step, cv_split, dataset_id, dataset_name, db_name, previous_step_id)

            # build pipeline list and save model identifier
            if fitted_step:
                pipeline_step = ("id_" + str(fitted_step.model_id), fitted_step.model)
                previous_step_id = fitted_step.model_id
            else:
                pipeline_step = ("n_" + str(i), copy.deepcopy(step.model))

            pipeline.append(pipeline_step)

            i += 1

        loaded_model = ModelWrapper(Pipeline(pipeline), model.fit_parameters)

    else:
        loaded_model = _load(model, cv_split, dataset_id, dataset_name, db_name)

        if not loaded_model:
            loaded_model = copy.deepcopy(model)

    return loaded_model


def fetch_fitted_models(db_name: str = "templates"):
    """
    Load fitted model from a database.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.datasets import make_classification
    >>> import lazygrid as lg
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
    >>> model = lg.ModelWrapper(model)
    >>>
    >>> save_model(model, cv_split, dataset_id, dataset_name, db_name)
    >>>
    >>> models = fetch_fitted_models(db_name)

    Parameters
    --------
    :param model: machine learning model
    :param cv_split: cross-validation split index
    :param dataset_id: data set identifier
    :param dataset_name: data set name
    :param db_name: database name
    :param previous_step_id: identifier of the previous step (for pipeline classifiers)
    :return: fitted model and model identifier (int)
    """

    # Connect to database if it exists
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
        return None

    stmt = '''SELECT id, type, fit_parameters, is_standalone, fitted_model, previous_step_id FROM MODEL'''
    models = cursor.execute(stmt).fetchall()

    # Load fitted model if present
    if not models:
        cursor.close()
        return None

    model_list = []
    for model in models:
        model_id, model_type, fit_parameters, is_standalone, model_bytes, previous_step_id = model

        if model_type in ["keras", "tensorflow"]:
            temp = os.path.join("./database", "temp.h5")
            with open(temp, 'wb') as output_file:
                output_file.write(model_bytes)
            fitted_model = keras.models.load_model(temp)
        else:
            fitted_model = pickle.loads(model_bytes)
        model_list.append(ModelWrapper(fitted_model, fit_parameters, model_id, bool(is_standalone)))

    db.close()

    return model_list


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

