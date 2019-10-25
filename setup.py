# -*- coding: utf-8 -*-
#
# Copyright 2019 - Barbiero Pietro and Squillero Giovanni
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

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='lazygrid',
    version='0.1.0',
    package_dir={'': 'lazygrid'},
    packages=find_packages(exclude=('tests', 'docs')),
    url='https://github.com/squillero/lazygrid-ng',
    license=license,
    author='Barbiero Pietro and Squillero Giovanni',
    author_email='cleisthenes.megacleos@gmail.com',
    description='LazyGrid: memoization of ML models',
    long_description=readme
)
