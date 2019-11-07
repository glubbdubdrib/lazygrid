import copy
import functools
import os
import pickle
import sys
from typing import Callable
import json
import keras
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from abc import ABC, abstractmethod, abstractproperty
import tensorflow as tf
from .neural_models import reset_weights
from .config import create_table_stmt, insert_model_stmt, query_stmt
from .database import save_model_to_db, load_model_from_db


class Wrapper(ABC):

    def __init__(self, model, fit_params, model_id,
                 is_standalone, cv_split, dataset_id, dataset_name,
                 db_name, **kwargs):

        self.model_id = model_id
        self.model_name = str(model.__class__.__name__).split(".")[0]
        self.model_type = str(model.__module__).split(".")[0]
        self.version = sys.modules[self.model_type].__version__

        self.is_fitted = False
        self.is_standalone = is_standalone
        self.fit_parameters = parse_fit_params(fit_params)
        self.models = None
        self.models_id = None

        self.model = model
        self.parameters = None
        self.serialized_model = None

        self.cv_split = cv_split
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.db_name = db_name
        self.previous_step_id = None

        self.entry = (
            self.model_name,
            self.model_type,
            pickle.dumps(self.__class__, protocol=2),
            str(self.parameters),
            self.fit_parameters,
            self.version,
            int(self.is_standalone),
            str(self.models_id),
            self.dataset_id,
            self.dataset_name,
            self.cv_split,
            self.previous_step_id,
            self.serialized_model,
        )

        self.query = (
            self.model_name,
            self.model_type,
            str(self.parameters),
            self.fit_parameters,
            self.version,
            str(self.models_id),
            self.dataset_id,
            self.cv_split,
            self.previous_step_id,
        )

    def to_database(self, **kwargs):
        save_model_to_db(self, create_table_stmt, insert_model_stmt, query_stmt)

    def from_database(self, **kwargs):
        return load_model_from_db(self, query_stmt)

    def load_model(self, **kwargs):
        result = self.from_database()
        if result:
            model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
            self.is_fitted = True
            self.model_id = model_id
            self.model = pickle.loads(serialized_model)
            self.serialized_model = serialized_model
        else:
            self.is_fitted = False
        return self

    def save_model(self, **kwargs):
        if not self.serialized_model:
            self.serialized_model = pickle.dumps(self.model, protocol=2)
        self.is_fitted = True
        self.to_database()

    def set_random_seed(self, **kwargs):
        raise NotImplementedError

    def parse_parameters(self, **kwargs):
        raise NotImplementedError

    def fit(self, **kwargs):
        if not self.is_fitted:
            self.model.fit(**kwargs)

    def predict(self, **kwargs):
        return self.model.predict(**kwargs)

    def score(self, **kwargs):
        return self.model.score(**kwargs)


class SklearnWrapper(Wrapper):
    """
    Class to wrap model objects (sklearn, keras, tensorflow).
    """

    def __init__(self, model, fit_params=None, model_id=None, is_standalone=True):
        Wrapper.__init__(self, model, fit_params, model_id, is_standalone)
        self.parameters = self.parse_parameters(model)

    def set_random_seed(self, seed: int):
        try:
            self.model.set_params(**{"random_state": seed})
        except AttributeError:
            pass

    def parse_parameters(self, model):
        return parse_sklearn_model(model)


class PipelineWrapper(Wrapper):
    """
    Class to wrap sklearn pipeline objects.
    """

    def __init__(self, model, fit_params, **kwargs):

        Wrapper.__init__(self, model, fit_params, **kwargs)
        self.models = []
        self.models_id = []
        for step in model.steps:
            pipeline_step = SklearnWrapper(step[1], fit_params, is_standalone=False)
            self.models.append(pipeline_step)
            self.models_id.append(pipeline_step.model_id)

    def save_model(self):
        for model in self.models:
            model.to_database()
        self.is_fitted = True
        self.to_database()

    def load_model(self):
        pipeline = []
        previous_step_id = -1
        i = 0
        for step in self.model.models:
            result = step.from_database(previous_step_id)
            if result:
                model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
                fitted_step = pickle.loads(serialized_model)
                pipeline_step = ("id_" + str(model_id), fitted_step)
                previous_step_id = model_id
            else:
                pipeline_step = ("n_" + str(i), step.model)
            pipeline.append(pipeline_step)
            i += 1
        return PipelineWrapper(Pipeline(pipeline), self.fit_parameters)

    def set_random_seed(self, seed: int):
        for parameter in list(self.model.get_params().keys()):
            if "random_state" in parameter:
                self.model.set_params(**{parameter: seed})

    def fit(self, x_train, y_train, **kwargs):
        x_train_t = x_train
        i = 0
        for pipeline_step, model in zip(self.model.steps, self.models):
            if not model.is_fitted:
                pipeline_step[1].fit(x_train_t, y_train)
                self.models[i] = SklearnWrapper(copy.deepcopy(pipeline_step[1]), is_standalone=False)
            if hasattr(pipeline_step[1], "transform"):
                x_train_t = pipeline_step[1].transform(x_train_t)
            i += 1


class KerasWrapper(Wrapper):
    """
    Class to wrap model objects (sklearn, keras, tensorflow).
    """

    def __init__(self, model, fit_params=None, model_id=None, is_standalone=True):

        Wrapper.__init__(self, model, fit_params, model_id, is_standalone)
        self.parameters = self.parse_parameters()

    def load_model(self, **kwargs):
        result = self.from_database()
        if result:
            model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
            self.is_fitted = True
            self.model_id = model_id
            self.model = load_neural_model(serialized_model)
            self.serialized_model = serialized_model
        else:
            self.is_fitted = False

    def save_model(self, **kwargs):
        if not self.serialized_model:
            self.serialized_model = save_neural_model(self.model)
        self.is_fitted = True
        self.to_database()

    def set_random_seed(self, seed: int):
        tf.set_random_seed(seed=seed)
        reset_weights(self.model, seed)

    def parse_parameters(self):
        return parse_neural_model(self.model)

    def fit(self, x, y, fit_params):
        self.model.fit(x, y, **fit_params)

    def predict(self, x, predict_params):
        return self.model.predict(x, **predict_params)

    def score(self, x, y, score_params):
        return self.model.score(x, y, **score_params)


def parse_sklearn_model(model):
    """
    Parse sklearn model parameters.

    :param model: sklearn model
    :return: None
    """

    args = []
    for attribute_name in dir(model):
        if not attribute_name.startswith("_") and not attribute_name.endswith("_"):
            try:
                attribute = getattr(model, attribute_name)
                if isinstance(attribute, Callable):
                    attribute = attribute.__name__
                else:
                    attribute = ' '.join(str(attribute).split())
                if attribute_name != str(attribute):
                    args.append(attribute_name + ": " + str(attribute))
            except (sklearn.exceptions.NotFittedError, AttributeError):
                pass
    return ", ".join(args)


def parse_neural_model(model):
    """
    Parse keras/tensorflow model parameters.

    :param model: neural network model
    :return: None
    """

    parameters = []
    parameters.append("input_shape: " + str(model.input_shape))
    parameters.append("output_shape: " + str(model.output_shape))

    layers = []
    trainable_layers = []
    for layer in model.layers:
        layers.append(layer.output_shape)
        if layer.trainable:
            trainable_layers.append(layer.output_shape)

    parameters.append("layers: " + str(layers))
    parameters.append("trainable_layers: " + str(trainable_layers))

    loss_functions = []
    for item in model.loss_functions:
        loss_functions.append(item.__name__)
    loss_functions.sort()
    parameters.append("loss_functions: " + str(loss_functions))

    losses = []
    for item in model.losses:
        losses.append(item.__name__)
    losses.sort()
    parameters.append("losses: " + str(losses))

    metrics = []
    for item in model.metrics:
        metrics.append(item)
    metrics.sort()
    parameters.append("metrics: " + str(metrics))

    metrics_names = []
    for item in model.metrics_names:
        metrics_names.append(item)
    metrics_names.sort()
    parameters.append("metrics_names: " + str(metrics_names))

    parameters.append("optimizer_class: " + model.optimizer.__class__.__name__)
    optimizer_parameters = []
    for attribute_name in dir(model.optimizer):
        attribute = getattr(model.optimizer, attribute_name)
        if type(attribute) is float:
            optimizer_parameters.append(str(attribute_name) + ": " + str(attribute))
    parameters.append("optimizer_params: " + str(optimizer_parameters))

    return str(parameters)


def corner_cases(model):
    if isinstance(model, RandomForestClassifier):
        if getattr(model, "n_estimators") == "warn":
            setattr(model, "n_estimators", 10)
    return model


def parse_fit_params(fit_params):
    if isinstance(fit_params, str):
        fit_params = json.loads(fit_params)
    # sort fit parameters dictionary
    if fit_params:
        fit_parameters = {}
        for param_key in sorted(fit_params):
            fit_parameters[param_key] = fit_params[param_key]
    else:
        fit_parameters = {}
    return json.dumps(fit_parameters)


def load_neural_model(model_bytes):
    temp = os.path.join("./tmp", "temp.h5")
    with open(temp, 'wb') as output_file:
        output_file.write(model_bytes)
    return keras.models.load_model(temp)


def save_neural_model(model):
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    temp = os.path.join("./tmp", "temp.h5")
    model.save(temp)
    with open(temp, 'rb') as input_file:
        fitted_model = input_file.read()
    return fitted_model
