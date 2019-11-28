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

import datetime
import logging
import os
from logging.handlers import RotatingFileHandler


log_info = logging.getLogger("lazygrid-debug")
log_info.addHandler(logging.NullHandler())
log_warn = logging.getLogger("lazygrid-warning")
log_warn.addHandler(logging.NullHandler())


def initialize_logging(path: str = "./log", log_name: str = "default-logger",
                       date: bool = True, output_console: bool = False) -> logging.Logger:
    """
    Initialize log file handler.

    Examples
    --------
    >>> import lazygrid as lg
    >>>
    >>> logger = lg.file_logger.initialize_logging()
    >>> logger.info("Log something")
    >>> lg.file_logger.close_logging(logger)

    Parameters
    ----------
    :param path: location where the log file will be saved
    :param log_name: log file name
    :param date: if True the current datetime will be put before the log file name
    :param output_console: if True it prints log messages to the default console
    :return: log file handler
    """

    if date:
        log_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + log_name
    log_name = os.path.join(path, log_name)

    # create log folder if it does not exists
    if not os.path.isdir(path):
        os.mkdir(path)

    # remove old logger if it exists
    if os.path.exists(log_name):
        os.remove(log_name)

    # log file format
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # create logger instances
    log_info.setLevel(logging.DEBUG)
    log_warn.setLevel(logging.WARNING)

    # the 'RotatingFileHandler' object implements a log file that is automatically limited in size
    # debug log handler
    fh_info = RotatingFileHandler(log_name + "-debug.log",
                                  mode='a',
                                  maxBytes=100 * 1024 * 1024,
                                  backupCount=2,
                                  encoding=None,
                                  delay=0)
    fh_info.setLevel(logging.DEBUG)
    fh_info.setFormatter(formatter)
    log_info.addHandler(fh_info)

    # warning log handler
    fh_warning = RotatingFileHandler(log_name + "-warning.log",
                                     mode='a',
                                     maxBytes=100 * 1024 * 1024,
                                     backupCount=2,
                                     encoding=None,
                                     delay=0)
    fh_warning.setLevel(logging.WARNING)
    fh_warning.setFormatter(formatter)
    log_warn.addHandler(fh_warning)

    if output_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log_info.addHandler(ch)

    log_info.debug("Starting " + log_name + "!")

    return log_info


def close_logging(logger: logging.Logger):
    """
    Close log file handler.

    Examples
    --------
    >>> import lazygrid as lg
    >>>
    >>> logger = lg.logger.initialize_logging()
    >>> logger.info("Log something")
    >>> lg.logger.close_logging(logger)

    Parameters
    ----------
    :param logger: log file handler
    :return: None
    """

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
