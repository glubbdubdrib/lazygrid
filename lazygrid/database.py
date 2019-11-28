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
from typing import Optional, Any, Iterable
import numpy as np
from .logger import log_warn


def save_to_db(db_name: str, entry: Iterable, query: Iterable,
               create_stmt: str, insert_stmt: str, query_stmt: str) -> Optional[Any]:
    """
    Save fitted model into a database.

    Examples
    --------
    >>> import lazygrid as lg
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> model = lg.wrapper.SklearnWrapper(LogisticRegression())
    >>>
    >>> entry = model.get_entry()
    >>> query = model.get_query()
    >>>
    >>> query_result = lg.database.save_to_db(model.db_name, entry, query)
    >>>
    >>> db_entry = lg.database.load_from_db(model.db_name, query)

    Parameters
    --------
    :param db_name: database name
    :param entry: collection that will be inserted into the database
    :param query: collection required to fetch a database entry
    :param create_stmt: database statement for table creation
    :param insert_stmt: database statement for model insertion
    :param query_stmt: database statement for model query
    :return: query result
    """

    # Sqlite does not accept INT larger than 8 bytes.
    sqlite3.register_adapter(np.int64, lambda val: int(val))
    sqlite3.register_adapter(np.int32, lambda val: int(val))

    # Create database if does not exists
    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, db_name + ".sqlite")
    db = sqlite3.connect(db_name)
    cursor = db.cursor()
    db.execute(create_stmt)

    # Insert item
    try:
        cursor.execute(insert_stmt, entry)
    except sqlite3.IntegrityError:
        log_warn.exception("Exception occurred")
        pass
    result = cursor.execute(query_stmt, query).fetchone()

    db.commit()
    db.close()

    return result


def load_from_db(db_name: str, query: Iterable, create_stmt: str, query_stmt: str) -> Optional[Any]:
    """
    Load fitted model from a database.

    Examples
    --------
    >>> import lazygrid as lg
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> model = lg.wrapper.SklearnWrapper(LogisticRegression())
    >>>
    >>> entry = model.get_entry()
    >>> query = model.get_query()
    >>>
    >>> query_result = lg.database.save_to_db(model.db_name, entry, query)
    >>>
    >>> db_entry = lg.database.load_from_db(model.db_name, query)

    Parameters
    --------
    :param db_name: database name
    :param query: collection required to fetch a database entry
    :param create_stmt: database statement for table creation
    :param query_stmt: database statement for model query
    :return: query result
    """

    # Sqlite does not accept INT larger than 8 bytes.
    sqlite3.register_adapter(np.int64, lambda val: int(val))
    sqlite3.register_adapter(np.int32, lambda val: int(val))

    # Connect to database if it exists
    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, db_name + ".sqlite")
    db = sqlite3.connect(db_name)
    cursor = db.cursor()
    db.execute(create_stmt)

    result = cursor.execute(query_stmt, query).fetchone()
    cursor.close()

    return result


def load_all_from_db(db_name: str, table_name: str = "MODEL") -> Optional[Any]:
    """
    Load all database items.

    Examples
    --------
    >>> from sklearn.linear_model import RidgeClassifier
    >>> from sklearn.datasets import make_classification
    >>> import lazygrid as lg
    >>>
    >>> x, y = make_classification(random_state=42)
    >>> model = lg.wrapper.SklearnWrapper(RidgeClassifier(),
    ...                                   db_name=db_name, dataset_id=1,
    ...                                   dataset_name="make-classification")
    >>> scores, _, _, _ = lg.model_selection.cross_validation(model, x, y)
    >>>
    >>> db_entries = lg.database.load_all_from_db(db_name)

    Parameters
    --------
    :param db_name: database name
    :param table_name: name of table to be loaded
    :return: query result
    """

    # Sqlite does not accept INT larger than 8 bytes.
    sqlite3.register_adapter(np.int64, lambda val: int(val))
    sqlite3.register_adapter(np.int32, lambda val: int(val))

    # Connect to database if it exists
    db_dir = "./database"
    if not os.path.isdir(db_dir):
        os.mkdir(db_dir)
    db_name = os.path.join(db_dir, db_name + ".sqlite")
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    stmt = '''SELECT name FROM sqlite_master WHERE type='table' AND name=? '''
    table = cursor.execute(stmt, (table_name,)).fetchone()
    if not table:
        db.close()
        return None

    query_stmt = '''SELECT * FROM %s''' % table_name
    result = cursor.execute(query_stmt).fetchall()
    cursor.close()

    return result


def drop_db(db_name: str) -> None:
    """
    Drop database table if it exists.

    Examples
    --------
    >>> import lazygrid as lg
    >>>
    >>> lg.database.drop_db(db_name="my-database")

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

