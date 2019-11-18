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
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        class BLOB NOT NULL,
        parameters TEXT NOT NULL,
        fit_parameters TEXT NOT NULL,
        predict_parameters TEXT NOT NULL,
        score_parameters TEXT NOT NULL,
        version TEXT NOT NULL,
        is_standalone INTEGER NOT NULL,
        models_id TEXT NOT NULL,
        dataset_id INTEGER NOT NULL,
        dataset_name TEXT,
        cv_split INTEGER NOT NULL,
        previous_step_id INTEGER,
        serialized_model BLOB NOT NULL,
        UNIQUE (name, type, parameters, 
                fit_parameters, predict_parameters, score_parameters,
                version, models_id, dataset_id, 
                cv_split, previous_step_id)
        )'''

insert_model_stmt = '''INSERT INTO MODEL(
        name, type, class, parameters, 
        fit_parameters, predict_parameters, score_parameters, 
        version, is_standalone, models_id,
        dataset_id, dataset_name, cv_split, previous_step_id, serialized_model)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

query_model_stmt = '''SELECT id, type, class, serialized_model, fit_parameters, is_standalone FROM MODEL
                      WHERE name=? AND type=? AND parameters=? AND 
                            fit_parameters=? AND predict_parameters=? AND score_parameters=? AND
                            version=? AND models_id=? AND dataset_id=? AND 
                            cv_split=? AND previous_step_id=?'''
