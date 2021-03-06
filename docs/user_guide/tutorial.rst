Tutorial
========

LazyGrid has three main features:

- it can generate all possible pipelines given a set of steps
  (`Pipeline generation <#pipeline-generation>`__) or all possible models
  given a grid of parameters (`Grid search <#grid-search>`__)
- it can compare the performance of a list of models using cross-validation
  and statistical tests (`Model comparison <#model-comparison>`__), and
- it follows the
  `memoization paradigm <https://en.wikipedia.org/wiki/Memoization>`__,
  avoiding fitting a model or a pipeline step twice.

Environment setup
-----------------

Input data
^^^^^^^^^^

In order to make each LazyPipeline transformer unique for different
cross-validation splits, you must provide input data as
`DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`__
objects. The easiest way to transform numpy arrays into ``DataFrame``
data structures is the following:

.. code:: python

    import pandas as pd
    ...
    X, y = ...
    X = pd.DataFrame(X)

Organizing data sets and databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using more than one data set in your project, it is highly
recommended to generate a hierarchy of database directories
so that models fitted on different data sets can be easily identified:

.. code:: python

    import os
    ...
    database_root_dir = "database"
    data_set_name = "foo"
    database_dir = os.path.join(database_root_dir, data_set_name)
    if not os.path.isdir(database_dir):
        os.makedirs(database_dir)

This code will generate a directory structure as the following:

.. code-block:: text

    database
    +-- foo
    |   +-- database.sqlite
    +-- baz
    |   +-- database.sqlite
    +-- ...



Model generation
----------------

Pipeline generation
^^^^^^^^^^^^^^^^^^^

In order to generate all possible pipelines given a set of steps, you
should define a list of elements, which in turn are lists of pipeline
steps, i.e. preprocessors, feature selectors, classifiers, etc. Each
step could be either a ``sklearn`` object or a ``keras`` model.

Once you have defined the pipeline elements, the ``generate_grid``
method will return a list of models of type
``lazygrid.lazy_estimator.LazyPipeline``.

The ``LazyPipeline`` class extends the ``sklearn.pipeline.Pipeline`` class
by providing an interface to SQLite databases.

.. code:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import RobustScaler, StandardScaler
    import lazygrid as lg

    preprocessors = [StandardScaler(), RobustScaler()]
    feature_selectors = [SelectKBest(score_func=f_classif, k=1), SelectKBest(score_func=f_classif, k=2)]
    classifiers = [RandomForestClassifier(random_state=42), SVC(random_state=42)]

    elements = [preprocessors, feature_selectors, classifiers]

    list_of_models = lg.grid.generate_grid(elements)

Grid search
^^^^^^^^^^^

LazyGrid implements a useful functionality to emulate the grid search
algorithm by generating all possible models given the model structure
and its parameters.

In this case, you should define a dictionary of arguments for the model
constructor and a dictionary of arguments for the fit method. The
``generate_grid_search`` method will return the list of all possible
models.

The following example illustrates how to use this functionality to
compare keras models with different optimizers and fit parameters.

.. code:: python

    import keras
    from keras import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from keras.utils import to_categorical
    from sklearn.metrics import f1_score
    from sklearn.datasets import load_digits
    from sklearn.model_selection import StratifiedKFold
    import lazygrid as lg
    import numpy as np
    import pandas as pd
    from keras.wrappers.scikit_learn import KerasClassifier


    # define keras model generator
    def create_keras_model(optimizer):

        kmodel = Sequential()
        kmodel.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=x_train.shape[1:]))
        kmodel.add(MaxPooling2D(pool_size=(2, 2)))
        kmodel.add(Flatten())
        kmodel.add(Dense(1000, activation='relu'))
        kmodel.add(Dense(n_classes, activation='softmax'))

        kmodel.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return kmodel


    # load data set
    X, y = load_digits(return_X_y=True)
    X = pd.DataFrame(X)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    list_of_splits = [split for split in skf.split(x, y)]
    train_index, val_index = list_of_splits[0]
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]
    x_train = np.reshape(x_train, (x_train.shape[0], 8, 8, 1))
    x_val = np.reshape(x_val, (x_val.shape[0], 8, 8, 1))
    n_classes = len(np.unique(y_train))
    if n_classes > 2:
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)


    # cast keras model into sklearn model
    kmodel = KerasClassifier(create_keras_model, verbose=1, epochs=0)

    # define all possible model parameters of the grid
    model_params = {"optimizer": ['SGD', 'RMSprop']}
    fit_params = {"epochs": [5, 10, 20], "batch_size": [10, 20]}

    # generate all possible models given the parameters' grid
    models, fit_parameters = lg.grid.generate_grid_search(kmodel, model_params, fit_params)


You will find the conclusion of this example in the
`plot section <#plot-your-results>`__.

Model comparison
----------------


Optimized cross-validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``LazyPipeline`` objects can be extremely useful
when a large number of machine learning pipelines need to be compared
through cross-validation techniques.

In fact, once a pipeline step has been fitted, LazyGrid saves the fitted
step into a `SQLite <https://www.sqlite.org/index.html>`__ database.
Therefore, should the step be required by another pipeline, LazyGrid
fetches the model that has already been fitted from the database.

This approach may boost the speed of time-consuming steps as recursive
feature elimination techniques, voting classifiers or deep neural networks.

.. code:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.datasets import make_classification
    import lazygrid as lg
    import pandas as pd

    X, y = make_classification(random_state=42)
    X = pd.DataFrame(X)

    preprocessors = [StandardScaler(), RobustScaler()]
    feature_selectors = [RFE(RandomForestClassifier, n_features_to_select=10),
                         SelectKBest(score_func=f_classif, k=10)]
    classifiers = [RandomForestClassifier(random_state=42), SVC(random_state=42)]

    elements = [preprocessors, feature_selectors, classifiers]

    models = lg.grid.generate_grid(elements)

    for model in models:
        scores = cross_validate(model, X, y, cv=10)



Statistical hypothesis tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have generated a list of models (or pipelines), LazyGrid
provides friendly APIs to compare models' performances by using a
cross-validation procedure and by analyzing the outcomes applying
statistical hypothesis tests.

You can collect the cross-validation scores into a single list
and call the ``find_best_solution`` method provided by LazyGrid. Such
method applies the following algorithm: it looks for the model having
the highest mean value over its cross-validation scores ("the best
model"); it compares the distribution of the scores of each model
against the distribution of the scores of the best model applying a
statistical hypothesis test.

You can customize the comparison by modifying the statistical hypothesis
test (it should be compatible with ``scipy.stats``) or the significance
level for the test.

.. code:: python

    ...
    scores = []
    for model in models:
        score = cross_validate(model, X, y, cv=10)
        scores.append(score["test_score"])

    best_idx, best_solutions_idx, pvalues = lg.statistics.find_best_solution(scores,
                                                                             test=mannwhitneyu,
                                                                             alpha=0.05)



Data set APIs
-------------

LazyGrid includes a set of easy-to-use APIs to fetch
`OpenML <https://www.openml.org/>`__ data sets (NB: OpenML has a
database of more than 20000 data sets).

The ``fetch_datasets`` method allows you to smartly handle such data
sets: it looks for OpenML data sets compliant with the requirements
specified; for such data sets, it fetches the characteristics of
their latest version; it saves in a local cache file the properties
of such data sets, so that experiments can be easily reproduced using
the same data sets and versions. You will find the list of downloaded
data sets inside ``./data/<datetime>-datalist.csv``.

The ``load_openml_dataset`` method can then be used to download the
required data set version.

.. code:: python

    import lazygrid as lg

    datasets = lg.datasets.fetch_datasets(task="classification", min_classes=2,
                                          max_samples=1000, max_features=10)

    # get the latest (or cached) version of the iris data set
    data_id = datasets.loc["iris"].did

    x, y, n_classes = lg.datasets.load_openml_dataset(data_id)
