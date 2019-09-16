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

# import machine_learning
import templates as tp
import lazygrid as lz

DATA_DIR = "./dump/"


def main():

    logger, folder_name = tp.initialize_logging(DATA_DIR,
                                                "lazygrid-ng",
                                                no_date=True)

    X_train, y_train, X_test, y_test = lz.generate_data()

    # setup LazyGrid
    grid = lz.LazyGrid(folder_name)
    grid.create_cube(lz.PIPELINES)

    # cross validation
    grid.cross_val_cube(X_train, y_train, logger)

    # blind test
    logger.info("Starting blind test!")
    grid.train_cube(X_train, y_train, logger)
    grid.test_cube(X_test, y_test, logger)

    # compute summary of the results
    grid.compute_results(save=True)

    tp.close_logging(logger)

    return


if __name__ == "__main__":
    main()
