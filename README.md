# LazyGrid

LazyGrid is a machine learning model comparator that follows 
the [memoization](https://en.wikipedia.org/wiki/Memoization) paradigm, i.e.
that is able to save fitted models and return them if required later.


#### Table Of Contents

* [Installation](#installation)
* [How to use](#how-to-use)
    * [Pipeline generation](#pipeline-generation)
    * [Grid search generation](#grid-search-generation)
    * [Model comparison](#model-comparison)
    * [Memoization: optimized cross-validation](#memoization-optimized-cross-validation)
    * [Plots](#plots)
    * [Automatic comparison](#automatic-comparison)
    * [Data sets APIs](#data-sets-apis)
* [Contributing to LazyGrid](CONTRIBUTING.md)

## Installation

You can install LazyGrid from [PyPI](https://pypi.org/project/lazygrid/):

```shell script
$ pip install lazygrid
```

Lazygrid is known to be working on Python 3.5 and above. 
The package is compatible with 
[scikit-learn 0.21](https://scikit-learn.org/stable/index.html),
[tensorflow 1.14](https://www.tensorflow.org/) and [Keras 2.2.4](https://keras.io/).

## How to use

LazyGrid has three main features:
* it can generate all possible pipelines given a set of steps
* it can compare the performance of a list of models using cross-validation and 
statistical tests
* it follows the memoization paradigm, avoiding fitting a model or a pipeline step twice

### Pipeline generation

In order to generate all possible pipelines given a set of steps, you should define
a list of elements, which in turn are lists of pipeline steps, i.e. preprocessors,
feature selectors, classifiers, etc. Each step could be either a `sklearn` object
or a `keras` model.

Once you have defined the pipeline elements, the `generate_grid` method will
return a list of models of type `sklearn.Pipeline`.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler, StandardScaler
import lazygrid as lg

preprocessors = [StandardScaler(), RobustScaler()]
feature_selectors = [SelectKBest(score_func=f_classif, k=1), SelectKBest(score_func=f_classif, k=2)]
classifiers = [RandomForestClassifier(random_state=42), SVC(random_state=42)]

elements = [preprocessors, feature_selectors, classifiers]

list_of_models = lg.generate_grid(elements)
```

### Grid search generation

LazyGrid implements a useful functionality to emulate the grid search algorithm 
by generating all possible models given the model structure and its parameters. 

In this case, you should define a dictionary of arguments for the model
constructor and a dictionary of arguments for the fit method.
The `generate_grid_search` method will return the list of all possible models.

The following example illustrates how to use this functionality to compare
keras models with different optimizers and fit parameters.

```python
import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
import lazygrid as lg
import numpy as np
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
x, y = load_digits(return_X_y=True)

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
models = lg.generate_grid_search(kmodel, model_params, fit_params)


# define scoring function for one-hot-encoded lables
def score_fun(y, y_pred):
    y = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y, y_pred, average="weighted")


# cross validation
for model in models:
    score, fitted_models = lg.cross_validation(model=model, x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                                               db_name="database", dataset_id=1, random_data=False,
                                               dataset_name="make-classification", n_splits=3, scoring=score_fun)
```

### Model comparison

Once you have generated a list of models (or pipelines), LazyGrid provides friendly
APIs to compare models' performances by using a cross-validation procedure and by
analyzing the outcomes applying statistical hypothesis tests.

First, you should define a classification task 
(e.g. `x, y = make_classification(random_state=42)`), define the set of models you
would like to compare (e.g. `model1 = LogisticRegression(random_state=42)`), and
call for each model the `cross_val_score` method provided by `sklearn`.

Finally, you can collect the cross-validation scores into a single list and call
the `find_best_solution` method provided by LazyGrid. Such method applies the following
algorithm:
 * it looks for the model having the highest mean value over its cross-validation scores
 ("the best model");
 * it compares the distribution of the scores of each model against
 the distribution of the scores of the best model applying a 
 [statistical hypothesis test](lazygrid/statistics.md).

You can customize the comparison
by modifying the statistical hypothesis test (it should be compatible with `scipy.stats`)
or the significance level for the test.

```python
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import lazygrid as lg
from scipy.stats import mannwhitneyu

x, y = make_classification(random_state=42)

model1 = LogisticRegression(random_state=42)
model2 = RandomForestClassifier(random_state=42)
model3 = RidgeClassifier(random_state=42)

score1 = cross_val_score(estimator=model1, X=x, y=y, cv=10)
score2 = cross_val_score(estimator=model2, X=x, y=y, cv=10)
score3 = cross_val_score(estimator=model3, X=x, y=y, cv=10)

scores = [score1, score2, score3]
best_idx, best_solutions_idx, pvalues = lg.find_best_solution(scores, test=mannwhitneyu, alpha=0.05)
```

### Memoization: optimized cross-validation

LazyGrid includes an optimized implementation of cross-validation (`cross_validation`), 
specifically devised when a huge number of machine learning pipelines need to be compared.

In fact, once a pipeline step has been fitted, LazyGrid saves the fitted model
into a [SQLite](https://www.sqlite.org/index.html) database.
Therefore, should the step be required by another pipeline, 
LazyGrid fetches the model that has already been fitted from the database. 

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.datasets import make_classification
import lazygrid as lg

x, y = make_classification(random_state=42)

preprocessors = [StandardScaler(), RobustScaler()]
feature_selectors = [SelectKBest(score_func=f_classif, k=1), SelectKBest(score_func=f_classif, k=2)]
classifiers = [RandomForestClassifier(random_state=42), SVC(random_state=42)]

elements = [preprocessors, feature_selectors, classifiers]

models = lg.generate_grid(elements)

for model in models:
    score, fitted_models = lg.cross_validation(model=model, x=x, y=y, 
                                               db_name="database", dataset_id=1, 
                                               dataset_name="make-classification")
```

### Plots

LazyGrid includes some standard features for presenting results as plots, among which
confusion matrixes and box plots.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import lazygrid as lg

x, y = make_classification(random_state=42)

model = LogisticRegression(random_state=42)
score, fitted_models = lg.cross_validation(model=model, x=x, y=y, 
                                           db_name="database", dataset_id=1, 
                                           dataset_name="make-classification")

conf_mat = lg.confusion_matrix_aggregate(fitted_models, x, y)
classes = ["P", "N"]
title = "Confusion matrix"
lg.plot_confusion_matrix(conf_mat, classes, "conf_mat.png", title)
``` 

### Automatic comparison

The `compare_models` method provides a friendly approach to compare a list of models:
 * it calls the `cross_validation` method for each model, automatically performing 
 the optimized cross-validation using the memoization paradigm;
 * it calls the `find_best_solution` method, applying a statistical test on the
 cross-validation results;
 * it returns a `Pandas.DataFrame` containing a summary of the results.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.datasets import make_classification
import lazygrid as lg

x, y = make_classification(random_state=42)

preprocessors = [StandardScaler(), RobustScaler()]
feature_selectors = [SelectKBest(score_func=f_classif, k=1), SelectKBest(score_func=f_classif, k=2)]
classifiers = [RandomForestClassifier(random_state=42), SVC(random_state=42)]

elements = [preprocessors, feature_selectors, classifiers]

models = lg.generate_grid(elements)

fit_params = []
for model in models:
    fit_params.append({})

results = lg.compare_models(models=models, x_train=x, y_train=y, params=fit_params,
                            dataset_id=1, dataset_name="make-classification", n_splits=10)
```

### Data sets APIs

LazyGrid includes a set of easy-to-use APIs to fetch [OpenML](https://www.openml.org/) 
data sets (NB: OpenML has a database of more than 20000 data sets).
 
The `fetch_datasets` method allows you to smartly handle such data sets:
* it looks for OpenML data sets compliant with the requirements specified;
* for such data sets, it fetches the characteristics of their latest version;
* it saves in a local cache file the properties of such data sets, so that
experiments can be easily reproduced using the same data sets and versions.

The `load_openml_dataset` method can then be used to download the required data set
version.

```python
import lazygrid as lg

datasets = lg.fetch_datasets(task="classification", min_classes=2, 
                             max_samples=1000, max_features=10)

# get the latest (or cached) version of the iris data set
data_id = datasets.loc["iris"].did

x, y, n_classes = lg.load_openml_dataset(data_id)
```

## Licence

Copyright 2019 Pietro Barbiero and Giovanni Squillero.

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
