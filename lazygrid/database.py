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
import numpy as np
from .config import create_model_stmt, insert_model_stmt, query_model_stmt


def save_to_db(db_name: str, entry: tuple, query: tuple,
               create_stmt: str = create_model_stmt,
               insert_stmt: str = insert_model_stmt,
               query_stmt: str = query_model_stmt) -> Optional[Any]:
    """
    Save fitted model into a database.

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
        # print(traceback.format_exc())
        pass
    result = cursor.execute(query_stmt, query).fetchone()

    db.commit()
    db.close()

    return result


def load_from_db(db_name: str, query: tuple,
                 create_stmt: str = create_model_stmt,
                 query_stmt: str = query_model_stmt) -> Optional[Any]:
    """
    Load fitted model from a database.

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

