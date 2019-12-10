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

create_model_stmt = '''CREATE TABLE IF NOT EXISTS MODEL(
        id INTEGER PRIMARY KEY,
        train TEXT NOT NULL,
        features TEXT NOT NULL,
        parameters TEXT NOT NULL,
        ids TEXT NOT NULL,
        estimator BLOB NOT NULL,

        UNIQUE (train, features, parameters, ids)
        )'''

insert_model_stmt = '''INSERT INTO MODEL(
        train, features, parameters, ids, estimator)
        VALUES(?, ?, ?, ?, ?)'''

query_model_stmt = '''SELECT * FROM MODEL
                      WHERE train=? AND features=? AND parameters=? AND ids=?'''
