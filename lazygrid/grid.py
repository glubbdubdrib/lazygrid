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
from typing import Tuple, List, Iterator, Iterable
import copy
from keras import Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler, StandardScaler
from itertools import chain, combinations, product
from .lazy_estimator import LazyPipeline


def _powerset(iterable: Iterable) -> Iterator:
    """
    Compute the powerset of a collection.

    Parameters
    ----------
    iterable
        Collection of items

    Returns
    -------
    Iterator
        Powerset of the collection
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def generate_grid(elements: list, lazy: bool = True, **kwargs) -> list:
    """
    Generate all possible combinations of sklearn Pipelines given the input steps.

    Parameters
    ----------
    elements
        List of elements used to generate the pipelines
    lazy
        If True it generates LazyPipelines objects;
        if False it generates standard sklearn Pipeline objects
    kwargs
        Keyword arguments to generate Pipeline objects

    Returns
    -------
    list
        List of pipelines

    Example
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.preprocessing import RobustScaler, StandardScaler
    >>> import lazygrid as lg
    >>>
    >>> preprocessors = [StandardScaler(), RobustScaler()]
    >>> feature_selectors = [SelectKBest(score_func=f_classif, k=1), SelectKBest(score_func=f_classif, k=2)]
    >>> classifiers = [RandomForestClassifier(random_state=42), SVC(random_state=42)]
    >>>
    >>> elements = [preprocessors, feature_selectors, classifiers]
    >>>
    >>> pipelines = lg.grid.generate_grid(elements)
    """

    assert isinstance(elements, list)
    assert all([isinstance(step, list) for step in elements])

    # generate all possible combinations of steps
    subsets = []
    for step_list in product(*elements):
        for steps in _powerset(step_list):
            if len(steps) > 0:
                if steps[-1] == step_list[-1]:
                    subsets.append(steps)

    pipelines = []
    for subset in subsets:
        # create list of tuples (step_name, step_object) to feed sklearn Pipeline
        i = 0
        steps = []
        for step in subset:
            steps.append(("step_" + str(i), copy.deepcopy(step)))
            i += 1

        if lazy:
            pipeline = LazyPipeline(steps, **kwargs)
        else:
            pipeline = Pipeline(steps, **kwargs)

        pipelines.append(pipeline)

    return pipelines


def generate_grid_search(model: KerasClassifier, model_params: dict,
                         fit_params: dict) -> Tuple[List[Model], List[dict]]:
    """
    Generate all possible combinations of models.

    Parameters
    ----------
    model
        Model architecture
    model_params
        Model parameters. For each key the dictionary should contain a list of possible values
    fit_params
        Fit parameters. For each key the dictionary should contain a list of possible values

    Returns
    -------
    Tuple
        Models and their corresponding fit parameters

    Example
    --------
    >>> import keras
    >>> from keras import Sequential
    >>> from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    >>> import lazygrid as lg
    >>> from keras.wrappers.scikit_learn import KerasClassifier
    >>>
    >>> # define keras model generator
    >>> def create_keras_model(input_shape, optimizer, n_classes):
    ...     kmodel = Sequential()
    ...     kmodel.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    ...     kmodel.add(MaxPooling2D(pool_size=(2, 2)))
    ...     kmodel.add(Flatten())
    ...     kmodel.add(Dense(1000, activation='relu'))
    ...     kmodel.add(Dense(n_classes, activation='softmax'))
    ...
    ...     kmodel.compile(loss=keras.losses.categorical_crossentropy,
    ...                    optimizer=optimizer, metrics=['accuracy'])
    ...     return kmodel
    >>>
    >>> # cast keras model into sklearn model
    >>> kmodel = KerasClassifier(create_keras_model)
    >>>
    >>> # define all possible model parameters of the grid
    >>> model_params = {"optimizer": ['SGD', 'RMSprop'], "input_shape": [(28, 28, 3)], "n_classes": [10]}
    >>> fit_params = {"epochs": [5, 10, 20], "batch_size": [10, 20]}
    >>>
    >>> # generate all possible models given the parameters' grid
    >>> models, fit_parameters = lg.grid.generate_grid_search(kmodel, model_params, fit_params)
    """

    models = []
    fit_parameters = []
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
        fit_parameters.append(fit_params_dict)

    return models, fit_parameters
