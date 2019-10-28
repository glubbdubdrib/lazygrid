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

import traceback
import numpy as np
import sys
from itertools import product
import copy

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
