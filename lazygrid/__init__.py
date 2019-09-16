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

from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn import svm, ensemble, linear_model
from sklearn import preprocessing
import copy
from .lazygrid import generate_data, LazyGrid

classifier_RF = copy.deepcopy(ensemble.RandomForestClassifier)
classifier_SVC = copy.deepcopy(svm.SVC)
k = 5

preprocessors = [
        preprocessing.StandardScaler(),
        preprocessing.RobustScaler()
]
feature_selectors = [
        RFE(estimator=classifier_RF(), n_features_to_select=k, step=1),
        SelectKBest(score_func=f_classif, k=k),
]
classifiers = [
        classifier_RF(random_state=42),
        classifier_SVC(random_state=42),
]


PIPELINES = [preprocessors, feature_selectors, classifiers]
