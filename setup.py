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

from setuptools import setup, find_packages
import pathlib
import os
from pypandoc import convert

# The directory containing this file
package_dir = pathlib.Path(__file__).parent

# with open(os.path.join(package_dir, 'README.md')) as f:
readme_text = convert((package_dir / "README.md").read_text(), "rst")

# with open(os.path.join(package_dir, 'LICENSE.txt')) as f:
license_text = convert((package_dir / 'LICENSE.txt').read_text(), "rst")

setup(
    name='lazygrid',
    version='0.2.1',
    description='LazyGrid: Efficient cross-validation and statistical tests of complex '
                'machine learning pipelines and neural networks',
    long_description=readme_text,
    long_description_content_type="text/x-rst",
    url='https://github.com/pietrobarbiero/lazygrid',
    author='Pietro Barbiero and Giovanni Squillero',
    author_email='cleisthenes.megacleos@gmail.com',
    license="Apache 2.0",
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
)
