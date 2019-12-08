LazyGrid
========

|Build|
|Coverage|

|Docs|
|Dependendencies|

|PyPI download total|
|PyPI license|


|PyPI-version|
|Language|

|Repo size|
|Open issues|

|Maintenance|
|Contributors|

|Followers|
|Stars|



.. |Build| image:: https://img.shields.io/travis/glubbdubdrib/lazygrid?label=Master%20Build&style=for-the-badge
    :alt: Travis (.org)
    :target: https://travis-ci.org/glubbdubdrib/lazygrid

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/glubbdubdrib/lazygrid?label=Test%20Coverage&style=for-the-badge
    :alt: Codecov
    :target: https://codecov.io/gh/glubbdubdrib/lazygrid

.. |Docs| image:: https://img.shields.io/readthedocs/lazygrid/latest?style=for-the-badge
    :alt: Read the Docs (version)
    :target: https://lazygrid.readthedocs.io/en/latest/

.. |Dependendencies| image:: https://img.shields.io/requires/github/glubbdubdrib/lazygrid?style=for-the-badge
    :alt: Requires.io
    :target: https://requires.io/github/glubbdubdrib/lazygrid/requirements/?branch=master

.. |Repo size| image:: https://img.shields.io/github/repo-size/glubbdubdrib/lazygrid?style=for-the-badge
    :alt: GitHub repo size
    :target: https://github.com/glubbdubdrib/lazygrid

.. |PyPI download total| image:: https://img.shields.io/pypi/dm/lazygrid?label=downloads&style=for-the-badge
    :alt: PyPI - Downloads
    :target: https://pypi.python.org/pypi/lazygrid/

.. |Open issues| image:: https://img.shields.io/github/issues/glubbdubdrib/lazygrid?style=for-the-badge
    :alt: GitHub issues
    :target: https://github.com/glubbdubdrib/lazygrid

.. |PyPI license| image:: https://img.shields.io/pypi/l/lazygrid.svg?style=for-the-badge
   :target: https://pypi.python.org/pypi/lazygrid/

.. |Followers| image:: https://img.shields.io/github/followers/glubbdubdrib?style=social
    :alt: GitHub followers
    :target: https://github.com/glubbdubdrib/lazygrid

.. |Stars| image:: https://img.shields.io/github/stars/glubbdubdrib/lazygrid?style=social
    :alt: GitHub stars
    :target: https://github.com/glubbdubdrib/lazygrid

.. |PyPI-version| image:: https://img.shields.io/pypi/v/lazygrid?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.python.org/pypi/lazygrid/

.. |Contributors| image:: https://img.shields.io/github/contributors/glubbdubdrib/lazygrid?style=for-the-badge
    :alt: GitHub contributors
    :target: https://github.com/glubbdubdrib/lazygrid

.. |Language| image:: https://img.shields.io/github/languages/top/glubbdubdrib/lazygrid?style=for-the-badge
    :alt: GitHub top language
    :target: https://github.com/glubbdubdrib/lazygrid

.. |Maintenance| image:: https://img.shields.io/maintenance/yes/2019?style=for-the-badge
    :alt: Maintenance
    :target: https://github.com/glubbdubdrib/lazygrid



LazyGrid is a python package providing an automatic, efficient and flexible
implementation of complex machine learning pipeline generation and cross-validation.

Before fitting a model or a pipeline step, LazyGrid checks inside an internal
SQLite database if the model has already been fitted. If the model is found,
it won't be fitted again.

Documentation for the
`latest stable version <https://lazygrid.readthedocs.io/en/latest/>`__
is available on ReadTheDocs.


Table Of Contents
------------------

-  `Getting Started <#getting-started>`__
-  `Documentation <#documentation>`__
-  `Running tests <#running-tests>`__
-  `Contributing <#contributing>`__
-  `Authors <#authors>`__
-  `Licence <#licence>`__

Getting Started
---------------

You can install LazyGrid along with all its dependencies from
`PyPI <https://pypi.org/project/lazygrid/>`__:

.. code:: bash

    $ pip install -r requirements.txt lazygrid

or from source code:

.. code:: bash

    $ git clone https://github.com/glubbdubdrib/lazygrid.git
    $ cd ./lazygrid
    $ pip install -r requirements.txt .

LazyGrid is known to be working on Python 3.5 and above. The package is
compatible with `scikit-learn
0.21 <https://scikit-learn.org/stable/index.html>`__ (and above), `tensorflow
1.14 <https://www.tensorflow.org/>`__ and `Keras
2.2.4 <https://keras.io/>`__.


Documentation
-------------

Documentation for the
`latest stable version <https://lazygrid.readthedocs.io/en/latest/>`__
is available on ReadTheDocs.


Running tests
-------------

You can run all unittests from command line by using python:

.. code:: bash

    $ python -m unittest discover

or coverage:

.. code:: bash

    $ coverage run -m unittest discover


Contributing
------------

Please read
`Contributing.md <https://github.com/glubbdubdrib/lazygrid/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull requests to us.


Authors
-------

* Pietro Barbiero - Mathematical engineer - `GitHub <https://github.com/pietrobarbiero>`__
* Giovanni Squillero - Professor of computer science at Politecnico di Torino - `GitHub <https://github.com/squillero>`__

Licence
-------

Copyright 2019 Pietro Barbiero and Giovanni Squillero.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
