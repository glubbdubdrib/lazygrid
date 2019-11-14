Welcome to LazyGrid
===================


|Build|
|Coverage|

|PyPI license|
|PyPI-version|



.. |Build| image:: https://img.shields.io/travis/glubbdubdrib/lazygrid?label=Master%20Build&style=for-the-badge
    :alt: Travis (.org)
    :target: https://travis-ci.org/glubbdubdrib/lazygrid

.. |Coverage| image:: https://img.shields.io/codecov/c/gh/glubbdubdrib/lazygrid?label=Test%20Coverage&style=for-the-badge
    :alt: Codecov
    :target: https://codecov.io/gh/glubbdubdrib/lazygrid

.. |Docs| image:: https://img.shields.io/readthedocs/lazygrid/latest?style=for-the-badge
    :alt: Read the Docs (version)
    :target: https://lazygrid.readthedocs.io/en/latest/

.. |PyPI license| image:: https://img.shields.io/pypi/l/lazygrid.svg?style=for-the-badge
   :target: https://pypi.python.org/pypi/lazygrid/

.. |PyPI-version| image:: https://img.shields.io/pypi/v/lazygrid?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.python.org/pypi/lazygrid/


LazyGrid is a python package providing an automatic, efficient and
flexible implementation of complex machine
learning pipeline generation and cross-validation.

Before fitting a model or a pipeline step, LazyGrid checks inside an internal
SQLite database if the model has already been fitted. If the model is found,
it won't be fitted again.

Quick start
-----------

You can install LazyGrid along with all its dependencies from
`PyPI <https://pypi.org/project/lazygrid/>`__:

.. code:: bash

    $ pip install -r requirements.txt lazygrid

Source
------

The source code and minimal working examples can be found on
`GitHub <https://github.com/glubbdubdrib/lazygrid>`__.


.. toctree::
    :caption: User Guide
    :maxdepth: 2

    user_guide/installation
    user_guide/tutorial
    user_guide/contributing
    user_guide/running_tests

.. toctree::
    :caption: API Reference
    :maxdepth: 2

    modules/database
    modules/datasets
    modules/file_logger
    modules/grid
    modules/model_selection
    modules/neural_models
    modules/plotter
    modules/statistics
    modules/wrapper


.. toctree::
    :caption: Copyright
    :maxdepth: 1

    user_guide/authors
    user_guide/licence


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`