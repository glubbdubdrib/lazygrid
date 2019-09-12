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
from .base import FeatureSelector


class RFE(FeatureSelector):
    """
    TODO: comment
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "best%d_%s" % (k, score_func.__name__)
        self.version = (1, 0)
        # TODO: local database with ranking
        self.feature_selector = feature_selection.SelectKBest(score_func=score_func, k=k)

    def fit(self, X, y, **fit_params):
        self.feature_selector.fit(X, y)

    def transform(self, X, y=None, **fit_params):
        return self.feature_selector.transform(X)