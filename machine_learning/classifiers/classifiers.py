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

import copy
# from sklearn import svm, ensemble, linear_model
from .base import Classifier

# TODO: use sklearn pipeline using grid searh


def make_classifier(classifier: object) -> Classifier:

    class BasicClassifier(Classifier):
        """
        TODO: comment
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = classifier.__name__
            self.version = (1, 0)
            classifier_copy = copy.deepcopy(classifier)
            self.classifier = classifier_copy

        def fit(self, X, y=None, **fit_params):
            self.classifier.fit(X)

        def transform(self, X, y=None, **fit_params):
            return self.classifier.transform(X)

    return BasicClassifier
