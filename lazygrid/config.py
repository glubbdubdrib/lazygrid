create_table_stmt = '''CREATE TABLE IF NOT EXISTS MODEL(
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        class BLOB NOT NULL,
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
        UNIQUE (name, type, parameters, fit_parameters, version, models_id, dataset_id, 
                cv_split, previous_step_id)
        )'''

insert_model_stmt = '''INSERT INTO MODEL(
        name, type, class, parameters, fit_parameters, version, is_standalone, models_id,
        dataset_id, dataset_name, cv_split, previous_step_id, serialized_model)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

query_stmt = '''SELECT id, type, class, serialized_model, fit_parameters, is_standalone FROM MODEL
                WHERE name=? AND type=? AND parameters=? AND fit_parameters=? AND 
                      version=? AND models_id=? AND dataset_id=? AND 
                      cv_split=? AND previous_step_id=?'''