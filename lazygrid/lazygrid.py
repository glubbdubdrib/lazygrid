# -*- coding: utf-8 -*-

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
from itertools import product

import pandas as pd
import scipy
from joblib import dump, load
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold

sys.path.append("./db_management")
from db_management import db_add_row, init_database, db_get_row

VERSION = (1, 0)


class Step(object):
    """
    TODO: comment
    """

    def __init__(self, step_id, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = step_id
        self.version = VERSION
        self.name = type(model).__name__
        self.step_class = type(model)
        self.model = model

    def is_computed(self, filename: str) -> bool:
        """
        TODO: comment
        """
        if not os.path.exists(filename):
            return False
        return True


class Pipeline(object):
    """
    TODO: comment
    """

    def __init__(self, pipeline_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.id = pipeline_id
        self.version = VERSION

        self.steps = {}

        self.accuracy = {}
        self.accuracy["train_cv"] = []
        self.accuracy["test_cv"] = []
        self.accuracy["train_blind"] = None
        self.accuracy["test_blind"] = None

        self.fitted_models = {}
        self.fitted_models["train_cv"] = []
        self.fitted_models["train_blind"] = []

        self.split_index_train = 0
        self.split_index_test = 0

    def update_split_index(self, process_type):

        if "train" in process_type:
            self.split_index_train += 1
        if "test" in process_type:
            self.split_index_test += 1


class LazyGrid(object):
    """
    TODO: comment
    """

    def __init__(self, experiment_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipelines = {}
        self.version = VERSION
        self.experiment_dir = experiment_dir
        self.results = None

    def create_cube(self, pipelines):
        """
        TODO: comment
        """

        # create LazyGrid cube of pipelines
        pipeline_id = 0
        for pipeline in product(*pipelines):
            self.pipelines[pipeline_id] = Pipeline(pipeline_id)

            # insert steps into each pipeline
            step_id = 0
            for model in pipeline:

                # create step object and add to the current pipeline
                self.pipelines[pipeline_id].steps[step_id] = \
                    Step(step_id, model)

                step_id += 1

            pipeline_id += 1

        return

    def _get_step_file(self, step, step_index,
                       process_type, pipeline, prev_filename):
        """
        TODO: comment
        """

        step_name = step.name

        # if this is the first step, then add path to filename
        # otherwise add the previous filename step
        if step_index == 0:
            filename = os.path.join(self.experiment_dir, step_name)
        else:
            filename = prev_filename.replace(".joblib", "") + \
                "_" + step_name

        # tag filename according to the task:
        # - cross validation / blind test
        # - training phase / testing phase
        file_type = "_blind"
        if "cv" in process_type:
            if "train" in process_type:
                file_type = "_cv_" + str(pipeline.split_index_train)
            if "test" in process_type:
                file_type = "_cv_" + str(pipeline.split_index_test)

        filename += file_type + ".joblib"

        return filename

    def _is_last_step(self, steps, step_index):
        """
        TODO: comment
        """

        if step_index == len(steps) - 1:
            return True
        return False

    def _process_cube(self, X, y, logger, process_type,
                      pool=None, threadLock=None):
        """
        TODO: comment
        """
        
        db = init_database("output")
        
        for pipeline in self.pipelines.values():

            logger.info("\tProcessing pipeline %d" % (pipeline.id))

            X_t = X
            step_index = 0
            filename = ""

            for step in pipeline.steps.values():

                logger.info("\tProcessing step %s" % (step.name))

                filename = self._get_step_file(step, step_index,
                                               process_type, pipeline,
                                               filename)

                # if the step was already computed, just load it!
                # otherwise, if the current task is about training
                # then fit the model and save it for the next time!

                returned_row = db_get_row(db, filename)

                if len(returned_row) == 1:
                    with open(filename, 'wb') as pickled_file:
                        pickled_file.write(returned_row[0][1])
                        pickled_file.close()
                    model = load(filename)
                    os.remove(filename)
                    
                elif "train" in process_type:
                    model = step.model
                    model.fit(X_t, y)
                    dump(model, filename)
                    db_add_row(db, filename.split('.')[0].split('/')[1], "./"+ filename)
                    pipeline.fitted_models[process_type].append(filename)
                    os.remove(filename)
                else:
                    continue

                # if this is *NOT* the last step of the pipeline,
                # the transform the input into the output
                if not self._is_last_step(pipeline.steps, step_index):
                    X_t = model.transform(X_t)

                step_index = step_index + 1

            # finally compute model score!
            accuracy = model.score(X_t, y)
            if "blind" in process_type:
                pipeline.accuracy[process_type] = accuracy
            else:
                pipeline.accuracy[process_type].append(accuracy)

            logger.info("\tAccuracy: %.4f" % (accuracy))

            pipeline.update_split_index(process_type)

        return

    def train_cube(self, X, y, logger, process_type="train_blind",
                   pool=None, threadLock=None):
        """
        TODO: comment
        """

        logger.info("Let's train!")
        self._process_cube(X, y, logger, process_type)

        return

    def test_cube(self, X, y, logger, process_type="test_blind",
                  pool=None, threadLock=None):
        """
        TODO: comment
        """

        logger.info("Let's test!")
        self._process_cube(X, y, logger, process_type)

        return

    def cross_val_cube(self, X, y, logger, n_splits=10,
                       pool=None, threadLock=None):
        """
        TODO: comment
        """

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=42)

        logger.info("Starting cross validation...")

        split_index = 0
        for train_index, test_index in skf.split(X, y):

            logger.info("Split %d" % (split_index))

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.train_cube(X_train, y_train, logger, "train_cv")
            self.test_cube(X_test, y_test, logger, "test_cv")

            split_index += 1

        logger.info("Cross validation ended!")

        return

    def compute_results(self, save=False):
        """
        TODO: comment
        """

        columns = [
                "pipeline_id",
                "steps",
                "train_acc_blind",
                "test_acc_blind",
                "train_avg_cv",
                "test_avg_cv",
                "train_stderr_cv",
                "test_stderr_cv",
        ]
        for i in range(0, self.pipelines[0].split_index_train-1):
            columns.append("train_acc_cv" + str(i))
            columns.append("test_acc_cv" + str(i))

        df = pd.DataFrame(columns=columns)

        for pipeline in self.pipelines.values():

            row = [pipeline.id]

            steps = []
            for step in pipeline.steps.values():

                steps.append(step.name)

            row.append(steps)
            row.append(pipeline.accuracy["train_blind"])
            row.append(pipeline.accuracy["test_blind"])

            train_avg = []
            test_avg = []
            for i in range(0, pipeline.split_index_train-1):
                train_avg.append(pipeline.accuracy["train_cv"][i])
                test_avg.append(pipeline.accuracy["test_cv"][i])
            row.append(scipy.mean(train_avg))
            row.append(scipy.mean(test_avg))
            row.append(scipy.stats.sem(train_avg))
            row.append(scipy.stats.sem(test_avg))

            for i in range(0, pipeline.split_index_train-1):
                row.append(pipeline.accuracy["train_cv"][i])
                row.append(pipeline.accuracy["test_cv"][i])

            df = df.append(pd.DataFrame([row], columns=columns),
                           ignore_index=True)

        df = df.set_index("pipeline_id")
        self.results = df

        if save:
            df.to_csv(os.path.join(self.experiment_dir, "results.csv"))

        return


def generate_data():
    """
    TODO: comment
    """

    X, y = datasets.make_classification(n_samples=200,
                                        n_features=50,
                                        n_informative=5,
                                        n_redundant=15,
                                        shuffle=True,
                                        random_state=42)

    skf = StratifiedKFold(n_splits=2, shuffle=True,
                          random_state=42)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test
