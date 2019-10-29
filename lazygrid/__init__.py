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

# Version of the lazygrid package
__version__ = "0.2.1"

from .file_logger import initialize_logging, close_logging
from .datasets import fetch_datasets, load_openml_dataset, load_npy_dataset
from .statistics import confidence_interval_mean_t, confusion_matrix_aggregate
from .neural_models import reset_weights, keras_classifier
from .grid import generate_grid, generate_grid_search
from .model_selection import compare_models, cross_validation, find_best_solution
from .plotter import plot_confusion_matrix, plot_boxplots
