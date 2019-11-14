Welcome to LazyGrid
===================

LazyGrid is a python package providing an automatic, efficient and
flexible implementation of complex machine
learning pipeline generation and cross-validation.

Before fitting a model or a pipeline step, LazyGrid checks inside an internal
SQLite database if the model has already been fitted. If the model is found,
it won’t be fitted again.

User Guide
~~~~~~~~~~

.. toctree::
    :maxdepth: 1

    user_guide/installation
    user_guide/tutorial
    user_guide/contributing
    user_guide/running_tests
    user_guide/authors
    user_guide/licence

API Reference
~~~~~~~~~~~~~

.. toctree::
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


Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`