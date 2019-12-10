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
import logging


def _save_to_db(db_name: str, entry: Iterable, query: Iterable,
                create_stmt: str, insert_stmt: str, query_stmt: str) -> Optional[Any]:
    """
    Save fitted model into a database.

    Parameters
    ----------
    db_name
        Database name
    entry
        Collection that will be inserted into the database
    query
        Collection required to fetch a database entry
    create_stmt
        Database statement for table creation
    insert_stmt
        Database statement for model insertion
    query_stmt
        Database statement for model query

    Returns
    -------
    Optional[Any]
        Query result
    """
    # # Sqlite does not accept INT larger than 8 bytes.
    # sqlite3.register_adapter(np.int64, lambda val: int(val))
    # sqlite3.register_adapter(np.int32, lambda val: int(val))

    # Connect to database if it exists
    root_dir = os.path.dirname(db_name)
    if not os.path.isdir(root_dir) and root_dir:
        os.makedirs(root_dir)
    db = sqlite3.connect(db_name)
    cursor = db.cursor()
    db.execute(create_stmt)

    # Insert item
    try:
        cursor.execute(insert_stmt, entry)
    except sqlite3.IntegrityError:
        # logging.exception("Exception occurred")
        pass
    result = cursor.execute(query_stmt, query).fetchone()

    db.commit()
    db.close()

    return result


def _load_from_db(db_name: str, query: Iterable, create_stmt: str, query_stmt: str) -> Optional[Any]:
    """
    Load fitted model from a database.

    Parameters
    ----------
    db_name
        Database name
    query
        Collection required to fetch a database entry
    create_stmt
        Database statement for table creation
    query_stmt
        Database statement for model query

    Returns
    -------
    Optional[Any]
        Query result
    """
    # # Sqlite does not accept INT larger than 8 bytes.
    # sqlite3.register_adapter(np.int64, lambda val: int(val))
    # sqlite3.register_adapter(np.int32, lambda val: int(val))

    # Connect to database if it exists
    root_dir = os.path.dirname(db_name)
    if not os.path.isdir(root_dir) and root_dir:
        os.makedirs(root_dir)
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
    ----------
    db_name
        Database name
    table_name
        Database table to load

    Returns
    -------
    Optional[Any]
        Query result
    """
    # # Sqlite does not accept INT larger than 8 bytes.
    # sqlite3.register_adapter(np.int64, lambda val: int(val))
    # sqlite3.register_adapter(np.int32, lambda val: int(val))

    # Connect to database if it exists
    root_dir = os.path.dirname(db_name)
    if not os.path.isdir(root_dir) and root_dir:
        os.makedirs(root_dir)
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

    Parameters
    ----------
    db_name
        Database name

    Returns
    -------
    None

    Examples
    --------
    >>> import lazygrid as lg
    >>>
    >>> lg.database.drop_db(db_name="my-database.sqlite")
    """
    root_dir = os.path.dirname(db_name)
    if not os.path.isdir(root_dir) and root_dir:
        os.makedirs(root_dir)
    db = sqlite3.connect(db_name)
    cursor = db.cursor()

    cursor.execute("DROP TABLE IF EXISTS MODEL")

    db.close()

    return

