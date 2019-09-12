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

from itertools import product
from joblib import dump, load
import os
from sklearn import datasets
import machine_learning

DATA_DIR = "dump"


def patch_filename(name: str) -> str:
    """Remove special characters"""
    return name


def is_computed(pipeline_step: machine_learning.PiplineStep) -> bool:
    filename = os.path.join(DATA_DIR, (pipeline_step.name))
    if not os.path.exists(filename):
        return False
    data, version = load(filename)
    if pipeline_step.version > version:
        return False
    return True


def generate_data():

    X, y = datasets.make_classification(n_samples=200,
                                        n_features=50,
                                        n_informative=5,
                                        n_redundant=15,
                                        shuffle=True,
                                        random_state=42)

    return X, y


def main():
    for pipeline in product(*machine_learning.PIPELINES):
        # check if the dump exists

        print("Checking pipeline: %s" % (pipeline,))
        for step_class in pipeline:
            if step_class is None:
                continue
            step = step_class()
            if is_computed(step):
                print("%s:%s already computed\n" % (step.name, step.version))
            else:
                print("%s:%s NOT already computed\n" % (step.name,
                                                        step.version))


if __name__ == "__main__":
    main()
