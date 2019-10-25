# -*- coding: utf-8 -*-
#
# Copyright 2019 - Barbiero Pietro and Squillero Giovanni
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

import sys

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import lazygrid
from threading import Lock
import logging
import numpy as np
import pandas as pd


def main():

    # initialize logging
    unique_name = "unique-name"
    path = "./log/"
    logger = lazygrid.initialize_logging(path, unique_name)
    logger.info("Hi, I am a program, starting now!")
    logger.info(logger.handlers)

    seed = 42
    np.random.seed(seed)
    openml = False

    # load openML data sets from a fixed point in time (if possible)
    if openml:
        data = lazygrid.fetch_datasets(task="classification", output_dir="./data",
                                       max_samples=1000, max_features=20, min_classes=3,
                                       update_data=False, logger=logger)
        dataset = data.iloc[-1]
        x, y, n_classes = lazygrid.load_openml_dataset(data_id=dataset.did, logger=logger)

    # load data set on-demand from openML
    else:
        db_name = "wine"
        x, y, n_classes = lazygrid.load_openml_dataset(dataset_name=db_name, logger=logger)
        dataset = pd.Series([db_name, 1], index=["db_name", "did"])

    # define pipeline elements
    preprocessors = [StandardScaler(), RobustScaler()]
    feature_selectors = [SelectKBest(score_func=f_classif, k=2), RFE(estimator=LogisticRegression())]
    classifiers = [RandomForestClassifier(), SVC(), LogisticRegression()]
    elements = [preprocessors, feature_selectors, classifiers]

    # generate models
    models = lazygrid.generate_grid(elements)

    # define fit parameters
    params = []
    for model in models:
        params.append({})

    # generate neural models
    models.append(lazygrid.keras_classifier([10, 5], x.shape[1:], n_classes, lr=0.001))
    models.append(lazygrid.keras_classifier([10, 5], x.shape[1:], n_classes, lr=0.01))
    params.append({"epochs": 200, "verbose": 1})
    params.append({"epochs": 100, "verbose": 0})

    # compare models
    results = lazygrid.compare_models(models=models, x_train=x, y_train=y, dataset_id=dataset.did,
                                      dataset_name=dataset.db_name, output_dir="./output",
                                      experiment_name=dataset.db_name, logger=logger, random_data=True,
                                      random_model=True, x_val=x, y_val=y, n_splits=10,
                                      params=params)

    lazygrid.close_logging(logger)

    return 0


if __name__ == "__main__":
    sys.exit(main())
