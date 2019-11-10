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
import sqlite3
from typing import Optional, Any
from .config import create_table_stmt, insert_model_stmt, query_stmt


def save_model_to_db(model, create_table_stmt: str = create_table_stmt,
                     insert_model_stmt: str = insert_model_stmt,
                     query_stmt: str = query_stmt) -> Optional[Any]:
    """
    Save fitted model into a database.

    Parameters
    --------
    :param model: machine learning model
    :param create_table_stmt: database statement for table creation
    :param insert_model_stmt: database statement for model insertion
    :param query_stmt: database statement for model query
    :return: query result
    """

    # Create database if does not exists
    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, model.db_name + ".sqlite")
    db = sqlite3.connect(db_name)
    cursor = db.cursor()
    db.execute(create_table_stmt)

    # Insert item
    try:
        cursor.execute(insert_model_stmt, model.entry)
    except sqlite3.IntegrityError:
        # print(traceback.format_exc())
        pass
    result = cursor.execute(query_stmt, model.query).fetchone()

    db.commit()
    db.close()

    return result


def load_model_from_db(model, query_stmt: str = query_stmt) -> Optional[Any]:
    """
    Load fitted model from a database.

    Parameters
    --------
    :param model: machine learning model
    :param query_stmt: database statement for model query
    :return: query result
    """

    # Connect to database if it exists
    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, model.db_name + ".sqlite")
    db = sqlite3.connect(db_name)
    cursor = db.cursor()
    stmt = '''SELECT name FROM sqlite_master WHERE type='table' AND name=? '''
    table = cursor.execute(stmt, ("MODEL",)).fetchone()
    if not table:
        db.close()
        return None

    result = cursor.execute(query_stmt, model.query).fetchone()
    cursor.close()

    return result


# TODO: resolve fetch models
# def fetch_fitted_models(db_name: str = "templates"):
#     """
#     Load fitted model from a database.
#
#     Examples
#     --------
#     >>> from sklearn.ensemble import RandomForestClassifier
#     >>> from sklearn.feature_selection import SelectKBest, f_classif
#     >>> from sklearn.datasets import make_classification
#     >>> import lazygrid as lg
#     >>>
#     >>> x, y = make_classification()
#     >>>
#     >>> fs = SelectKBest(f_classif, k=5)
#     >>> clf = RandomForestClassifier()
#     >>> model = Pipeline([('feature_selector', fs), ('clf', clf)])
#     >>> type(model.fit(x, y))
#     <class 'sklearn.pipeline.Pipeline'>
#     >>>
#     >>> cv_split = 0
#     >>> dataset_id = 1
#     >>> dataset_name = "iris"
#     >>> fit_params = {}
#     >>> db_name = "templates"
#     >>> model = lg.ModelWrapper(model)
#     >>>
#     >>> save_model(model, cv_split, dataset_id, dataset_name, db_name)
#     >>>
#     >>> models = fetch_fitted_models(db_name)
#
#     Parameters
#     --------
#     :param model: machine learning model
#     :param cv_split: cross-validation split index
#     :param dataset_id: data set identifier
#     :param dataset_name: data set name
#     :param db_name: database name
#     :param previous_step_id: identifier of the previous step (for pipeline classifiers)
#     :return: fitted model and model identifier (int)
#     """
#
#     # Connect to database if it exists
#     db_dir = "./database"
#     if not os.path.isdir(db_dir):
#         os.mkdir(db_dir)
#     db_name = os.path.join(db_dir, db_name + ".sqlite")
#     db = sqlite3.connect(db_name)
#     cursor = db.cursor()
#     stmt = '''SELECT name FROM sqlite_master WHERE type='table' AND name=? '''
#     table = cursor.execute(stmt, ("MODEL",)).fetchone()
#     if not table:
#         db.close()
#         return None
#
#     stmt = '''SELECT id, type, class, fit_parameters, is_standalone, fitted_model, previous_step_id FROM MODEL'''
#     models = cursor.execute(stmt).fetchall()
#
#     # Load fitted model if present
#     if not models:
#         cursor.close()
#         return None
#
#     model_list = []
#     for m in models:
#         model_id, model_type, model_class, fit_parameters, is_standalone, model_bytes, previous_step_id = m
#         model_class = pickle.loads(model_class)
#
#         fitted_model = model_class.unpickle_model(model_type, model_bytes)
#         model = model_class(fitted_model, fit_parameters, model_id, is_standalone)
#         model_list.append(model)
#
#     db.close()
#
#     return model_list


def drop_db(db_name) -> None:
    """
    Drop database table if it exists.

    Examples
    --------
    >>> import os
    >>> import sqlite3
    >>>
    >>> drop_db()

    Parameters
    --------
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

