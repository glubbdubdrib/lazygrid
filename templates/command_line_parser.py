# -*- coding: utf-8 -*-
#
##############################
#                            #
# Command line parser module #
#                            #
##############################
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


import argparse
import sys


def parse_command_line():

    description = "Python script that ...\n" + \
                  "By [Authors], [years] [<email>]"
    parser = argparse.ArgumentParser(description=description)

    # required argument
    parser.add_argument("-b",
                        "--blabla",
                        help="File containing ...).",
                        required=True)
#
#    # list of elements, type int
#    parser.add_argument("-bla",
#                        "--blabla",
#                        help="File containing ...).",
#                        required=True,
#                        type=int,
#                        nargs='+')
#
#    # flag, it's just true/false
#    parser.add_argument("-bla",
#                        "--blabla",
#                        help="File containing ...).",
#                        action="store_true")

    args = parser.parse_args()

    return args


def main():

    # get command-line arguments
    args = parse_command_line()

    print(args)

    return


if __name__ == "__main__":
    sys.exit(main())
