# -*- coding: utf-8 -*-
#
##################
#                #
# Logging module #
#                #
##################
#
# Copyright 2019 Pietro Barbiero, Giovanni Squillero and Alberto Tonda
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

import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def initialize_logging(path=None, uniqueName=None, no_date=False):

    if no_date:
        folderName = "experiment"
    else:
        folderName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if uniqueName is not None:
        folderName += "-" + uniqueName
    if path is not None:
        if os.path.exists(path):
            folderName = path + folderName
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    logger = logging.getLogger("")

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s",
                                  "%Y-%m-%d %H:%M:%S")

    # the 'RotatingFileHandler' object implements a log file
    # that is automatically limited in size
    if uniqueName is not None:
        fh = RotatingFileHandler(os.path.join(folderName, "log.log"),
                                 mode='a',
                                 maxBytes=100*1024*1024,
                                 backupCount=2,
                                 encoding=None,
                                 delay=0)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if uniqueName is not None:
        logger.info("Starting " + uniqueName + "!")

    return logger, folderName


def close_logging(logger):

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    return


def main():

    # unique name of the directory
    uniqueName = "unique-name"

    # initialize logging, using a logger that smartly manages disk occupation
    logger = initialize_logging(uniqueName)

    # start program
    logger.info("Hi, I am a program, starting now!")
    logger.info(type(logger))
    close_logging(logger)

    return


if __name__ == "__main__":
    sys.exit(main())
