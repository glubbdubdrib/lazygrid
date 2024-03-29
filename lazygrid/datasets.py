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

import datetime
import glob
import traceback
from difflib import SequenceMatcher
import numpy as np
import openml
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import logging


def load_npy_dataset(path_x: str, path_y: str) -> (np.ndarray, np.ndarray, int):
    """
    Load npy data set.

    Parameters
    ----------
    path_x
        Path to data matrix
    path_y
        Path to data labels

    Returns
    -------
    Tuple
        Data matrix, data labels, and number of classes

    Examples
    --------
    >>> import os
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> import lazygrid as lg
    >>>
    >>> x, y = make_classification(random_state=42)
    >>>
    >>> path_x, path_y = "x.npy", "y.npy"
    >>> np.save(path_x, x)
    >>> np.save(path_y, y)
    >>>
    >>> x, y, n_classes = lg.datasets.load_npy_dataset(path_x, path_y)
    """
    try:
        x = np.load(path_x)
        y = np.load(path_y)
        n_classes = len(np.unique(y))

    except FileNotFoundError:
        logging.exception("Exception occurred")
        return None, None, None

    return x, y, n_classes


def load_openml_dataset(data_id: int = None, dataset_name: str = None) -> (np.ndarray, np.ndarray, int):
    """
    Load OpenML data set.

    Parameters
    ----------
    data_id
        Data set identifier
    dataset_name
        Data set name

    Returns
    -------
    Tuple
        Data matrix, data labels, and number of classes

    Examples
    --------

    >>> import lazygrid as lg
    >>>
    >>> x, y, n_classes = lg.datasets.load_openml_dataset(dataset_name="iris")
    >>> n_classes
    3
    """
    assert (data_id is not None) or (dataset_name is not None)
    assert isinstance(dataset_name, str) or dataset_name is None

    try:
        if data_id is not None:
            x, y = fetch_openml(data_id=data_id, return_X_y=True)
        else:
            x, y = fetch_openml(name=dataset_name, return_X_y=True)

        if isinstance(x, pd.DataFrame):
            x = x.values

        if not isinstance(x, np.ndarray):
            x = x.toarray()

        si = SimpleImputer(missing_values=np.nan, strategy='mean')
        x = si.fit_transform(x)

        le = LabelEncoder()
        y = le.fit_transform(y)
        n_classes = np.unique(y).shape[0]

        return x, y, n_classes

    except Exception:
        logging.exception("Exception occurred")
        return [None, None, None]


# def _similar(a: str, b: str) -> float:
#     """
#     Compute how much two strings are similar.
#     """
#     return SequenceMatcher(None, a, b).ratio()


def _is_correct_task(task: str, db: dict) -> bool:
    """
    Check if the current data set is compatible with the specified task.

    Parameters
    ----------
    task
        Regression or classification
    db
        OpenML data set dictionary

    Returns
    -------
    bool
        True if the task and the data set are compatible
    """
    if task == "classification":
        return db['NumberOfSymbolicFeatures'] == 1 and db['NumberOfClasses'] > 0
    elif task == "regression":
        return True
    else:
        return False


def _load_datasets(output_dir: str = "./data", min_classes: int = 0, task: str = "classification",
                   max_samples: int = np.inf, max_features: int = np.inf) -> pd.DataFrame:
    """
    Load all OpenML data sets compatible with the requirements and save a .csv file as a reference.

    Parameters
    ----------
    output_dir
        Directory where the .csv file will be stored
    min_classes
        Minimum number of classes required for each data set
    task
        Classification or regression
    max_samples
        Maximum number of samples required for each data set
    max_features
        Maximum number of features required for each data set

    Returns
    -------
    Dataframe
        Information required to load the latest version of each data set

    Examples
    --------
    >>> datasets = _load_datasets(task="classification", min_classes=2, max_samples=1000, max_features=10)
    >>> datasets.loc["iris"]
    version          45
    did           42098
    n_samples       150
    n_features        4
    n_classes         3
    Name: iris, dtype: int64
    """

    data = {}

    for key, db in openml.datasets.list_datasets().items():

        try:
            logging.info("Loading data set: %s, ID: %d..." % (db['name'], db['did']))

            if db['NumberOfClasses'] > min_classes and _is_correct_task(task, db) and \
                    db['NumberOfInstances'] < max_samples and db['NumberOfFeatures'] < max_features and \
                    db['status'] == 'active':

                if db['name'] in data:
                    if db['version'] < data[db['name']]['version']:
                        continue

                # load_openml_dataset(db['did'])

                data[db['name']] = {}
                data[db['name']]['version'] = db['version']
                data[db['name']]['did'] = db['did']
                data[db['name']]['n_samples'] = db['NumberOfInstances']
                data[db['name']]['n_features'] = db['NumberOfNumericFeatures']
                data[db['name']]['n_classes'] = db['NumberOfClasses']

        except (IndexError, ValueError, KeyError):
            logging.info("Error loading data set: %s, ID: %d!" % (db['name'], db['did']))
            logging.info(traceback.format_exc())

    data = pd.DataFrame(data).transpose()
    try:
        data = data.sort_values(by=["n_samples", "n_features", "n_classes"]).astype('int64')
    except KeyError:
        logging.info(traceback.format_exc())
        return pd.DataFrame()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-datalist.csv"
    data.to_csv(os.path.join(output_dir, file_name), index_label="db_name")

    return data


def fetch_datasets(output_dir: str = "./data", update_data: bool = False,
                   min_classes: int = 0, task: str = "classification",
                   max_samples: int = np.inf, max_features: int = np.inf) -> pd.DataFrame:
    """
    Load OpenML data sets compatible with the requirements.

    Parameters
    ----------
    output_dir
        Directory where the .csv file will be stored
    update_data
        If True it deletes cached data sets and downloads their latest version;
        otherwise it loads data sets as specified inside the cache
    min_classes
        Minimum number of classes required for each data set
    task
        Classification or regression
    max_samples
        Maximum number of samples required for each data set
    max_features
        Maximum number of features required for each data set

    Returns
    -------
    Dataframe
        Information required to load the latest version of each data set

    Examples
    --------
    >>> import lazygrid as lg
    >>>
    >>> datasets = lg.datasets.fetch_datasets(task="classification", min_classes=2, max_samples=1000, max_features=10)
    >>> datasets.loc["iris"]
    version          45
    did           42098
    n_samples       150
    n_features        4
    n_classes         3
    Name: iris, dtype: int64
    """
    files_location = os.path.join(output_dir, '*.csv')
    file_list = glob.glob(files_location)

    data = None

    # download (again) new data if necessary
    if not os.path.isdir(output_dir) or not file_list or update_data:
        data = _load_datasets(output_dir, min_classes, task, max_samples, max_features)

        # delete previous data files
        file_list.sort()
        for file in file_list[:-1]:
            os.remove(file)

    # fetch last version of downloaded data if there is one
    elif file_list:
        file_list = glob.glob(files_location)
        file_list.sort()
        data = pd.read_csv(file_list[-1], index_col=0).astype("int64")

    return data
