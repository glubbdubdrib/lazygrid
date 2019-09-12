# -*- coding: utf-8 -*-

# Copyright 2019 Giovanni Squillero and Pietro Barbiero
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

from sklearn import feature_selection
from sklearn import svm, ensemble, linear_model
from .generic_step import PiplineStep
from .pre_processors.scalers import StandardScaler
from .feature_selectors.k_best import make_kbest
from .classifiers.classifiers import make_classifier

preprocessors = [StandardScaler]
feature_selectors = [
        make_kbest(k=100,
                   method_parameters={"method": feature_selection.SelectKBest,
                                      "helper": feature_selection.f_classif}),

        make_kbest(k=10,
                   method_parameters={"method": feature_selection.SelectKBest,
                                      "helper": feature_selection.f_classif}),

        make_kbest(k=10,  method_parameters={"method": feature_selection.RFE,
                                             "helper": svm.SVC,
                                             "step": 1}),
]
classifiers = [
        make_classifier(svm.SVC),
        make_classifier(ensemble.RandomForestClassifier),
        make_classifier(svm.SVC),
]

PIPELINES = [preprocessors, feature_selectors, classifiers]
