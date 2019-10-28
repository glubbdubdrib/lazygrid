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
import functools
import traceback
from abc import ABCMeta
from typing import Union

import numpy as np
import sys
from itertools import product
import copy

from keras import Sequential, Model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.datasets import make_classification
from sklearn.preprocessing import RobustScaler, StandardScaler


def generate_grid(elements: list) -> list:
    """
    Generate all possible combinations of sklearn Pipelines given the input steps.

    Example
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.preprocessing import RobustScaler, StandardScaler
    >>>
    >>> preprocessors = [StandardScaler(), RobustScaler()]
    >>> feature_selectors = [SelectKBest(score_func=f_classif, k=1), SelectKBest(score_func=f_classif, k=2)]
    >>> classifiers = [RandomForestClassifier(random_state=42), SVC(random_state=42)]
    >>>
    >>> elements = [preprocessors, feature_selectors, classifiers]
    >>>
    >>> pipelines = generate_grid(elements)

    Parameters
    --------
    :param elements: list of elements used to generate the pipelines. It should be a list of N lists,
                     each one containing an arbitrary number of elements of the same kind.
    :return: list of sklearn Pipelines
    """

    assert isinstance(elements, list)
    assert all([isinstance(step, list) for step in elements])

    # generate all possible combinations of steps
    pipelines = []
    for step_list in product(*elements):

        # create list of tuples (step_name, step_object) to feed sklearn Pipeline
        i = 0
        steps = []
        for step in step_list:
            steps.append(("step_" + str(i), copy.deepcopy(step)))
            i += 1
        pipeline = Pipeline(steps)

        pipelines.append(pipeline)

    return pipelines


def generate_grid_search(model: Union[Sequential, Model, ABCMeta, Pipeline],
                         model_params: dict, fit_params: dict) -> list:
    """
    Generate all possible combinations of models.

    Parameters
    --------
    :param model: model architecture
    :param model_params: model parameters
    :param fit_params: fit parameters
    :return: list of sklearn Pipelines
    """

    models = []
    keys = []
    values = []
    is_for_fit = []

    # fill keys' and values' lists with model parameters
    for key_model, value_model in model_params.items():
        keys.append(key_model)
        values.append(value_model)
        is_for_fit.append(False)

    # fill keys' and values' lists with fit parameters
    for key_fit, value_fit in fit_params.items():
        keys.append(key_fit)
        values.append(value_fit)
        is_for_fit.append(True)

    # generate all possible combinations of parameters
    for values_list in product(*values):

        learner = copy.deepcopy(model)

        # uniquely define model structure
        model_params_values = [values_list[i] for i in range(0, len(values_list)) if is_for_fit[i] is False]
        model_params_keys = [keys[i] for i in range(0, len(keys)) if is_for_fit[i] is False]
        model_params_dict = dict(zip(model_params_keys, model_params_values))
        learner = learner.build_fn(**model_params_dict)

        # uniquely define fit function
        fit_params_values = [values_list[i] for i in range(0, len(values_list)) if is_for_fit[i] is True]
        fit_params_keys = [keys[i] for i in range(0, len(keys)) if is_for_fit[i] is True]
        fit_params_dict = dict(zip(fit_params_keys, fit_params_values))
        functools.partial(learner.fit, **fit_params_dict)

        models.append(learner)

    return models
