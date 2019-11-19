import copy
import os
import pickle
import sys
from typing import Callable, Any, Union
import json
import keras
import sklearn
from sklearn.ensemble import RandomForestClassifier
from abc import ABC
import tensorflow as tf
from .database import drop_db, save_to_db, load_from_db
from .neural_models import reset_weights


class Wrapper(ABC):
    """
    Abstract wrapper class for machine learning models.
    """

    def __init__(self, model: Any, dataset_id=None, dataset_name=None, db_name=None,
                 fit_params=None, predict_params=None, score_params=None, model_id=None,
                 cv_split=None, is_standalone=True, **kwargs):

        """
        Wrapper initialization.

        Parameters
        --------
        :param model: machine learning model
        :param dataset_id: data set identifier
        :param dataset_name: data set name
        :param db_name: database name
        :param fit_params: model's fit parameters
        :param predict_params: model's predict parameters
        :param score_params: model's score parameters
        :param model_id: model identifier
        :param cv_split: cross-validation split identifier
        :param is_standalone: True if model can be used independently from other models
        :param kwargs: other parameters
        """

        if not dataset_id:
            db_name = None
            dataset_id = 1
            dataset_name = "default-dataset"
        if not db_name:
            db_name = "default-db"
            drop_db(db_name)

        self.model_id = model_id
        self.model_name = str(model.__class__.__name__).split(".")[0]
        self.model_type = str(model.__module__).split(".")[0]
        self.version = sys.modules[self.model_type].__version__

        self.is_fitted = False
        self.is_standalone = is_standalone
        self.fit_parameters = dict_to_json(fit_params)
        self.predict_parameters = dict_to_json(predict_params)
        self.score_parameters = dict_to_json(score_params)
        self.models = None
        self.models_id = None

        self.model = model
        self.parameters = None
        self.serialized_model = None

        self.cv_split = cv_split
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.db_name = db_name
        self.previous_step_id = -1

        self.entry = None
        self.query = None

    def get_entry(self) -> tuple:
        """
        Define database entry for the model.

        Parameters
        --------
        :return: database entry
        """

        entry = (
            self.model_name,
            self.model_type,
            pickle.dumps(self.__class__, protocol=2),
            str(self.parameters),
            self.fit_parameters,
            self.predict_parameters,
            self.score_parameters,
            self.version,
            int(self.is_standalone),
            str(self.models_id),
            self.dataset_id,
            self.dataset_name,
            self.cv_split,
            self.previous_step_id,
            self.serialized_model,
        )
        return entry

    def get_query(self) -> tuple:
        """
        Define database query for the model.

        Parameters
        --------
        :return: database query
        """
        query = (
            self.model_name,
            self.model_type,
            str(self.parameters),
            self.fit_parameters,
            self.predict_parameters,
            self.score_parameters,
            self.version,
            str(self.models_id),
            self.dataset_id,
            self.cv_split,
            self.previous_step_id,
        )
        return query

    def to_database(self, **kwargs):
        """
        Send model into a database.

        Parameters
        --------
        :param kwargs: some parameters
        :return: query result
        """
        self.entry = self.get_entry()
        self.query = self.get_query()
        return save_to_db(self.db_name, self.entry, self.query)

    def from_database(self, **kwargs):
        """
        Fetch model from a database.

        Parameters
        --------
        :param kwargs: some parameters
        :return: query result
        """
        self.query = self.get_query()
        return load_from_db(self.db_name, self.query)

    def load_model(self, **kwargs) -> Any:
        """
        Load model from database if possible.

        Parameters
        --------
        :param kwargs: some parameters
        :return: wrapped model
        """
        result = self.from_database()
        if result:
            model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
            self.is_fitted = True
            self.model_id = model_id
            self.model = pickle.loads(serialized_model)
            self.serialized_model = serialized_model
        return self

    def save_model(self, **kwargs) -> None:
        """
        Save model into database.

        Parameters
        --------
        :param kwargs: some parameters
        :return: None
        """
        if not self.serialized_model:
            self.serialized_model = pickle.dumps(self.model, protocol=2)
        result = self.to_database()
        if result:
            model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
            self.model_id = model_id

    def set_random_seed(self, seed, split_index, random_model, **kwargs):
        """
        Set model random state if possible.

        Parameters
        --------
        :param seed: random seed
        :param split_index: cross-validation split identifier
        :param random_model: whether the model should have the same random state for each cross-validation split
        :param kwargs: some parameters
        :return: None
        """
        raise NotImplementedError

    def parse_parameters(self, **kwargs) -> str:
        """
        Parse model parameters.

        Parameters
        --------
        :param kwargs: some parameters
        :return: parameters as a string
        """
        raise NotImplementedError

    def fit(self, x, y, **kwargs) -> None:
        """
        Fit model with some samples.

        Parameters
        --------
        :param x: train data
        :param y: train labels
        :param kwargs: some parameters
        :return: None
        """
        if not self.is_fitted:
            self.model.fit(x, y)
        self.is_fitted = True

    def predict(self, x, **kwargs) -> Any:
        """
        Predict labels for some input samples.

        Parameters
        --------
        :param x: input data
        :param kwargs: some parameters
        :return: predictions
        """
        return self.model.predict(x)

    def score(self, x, y, **kwargs) -> Any:
        """
        Compute score for some input samples.

        Parameters
        --------
        :param x: input data
        :param y: input labels
        :param kwargs: some parameters
        :return: score
        """
        return self.model.score(x, y)


class SklearnWrapper(Wrapper):
    """
    Class to wrap sklearn models.
    """

    def __init__(self, model: Any, dataset_id=None, dataset_name=None, db_name=None,
                 fit_params=None, predict_params=None, score_params=None, model_id=None,
                 cv_split=None, is_standalone=True, **kwargs):
        """
        Wrapper initialization.

        Parameters
        --------
        :param model: machine learning model
        :param dataset_id: data set identifier
        :param dataset_name: data set name
        :param db_name: database name
        :param fit_params: model's fit parameters
        :param predict_params: model's predict parameters
        :param score_params: model's score parameters
        :param model_id: model identifier
        :param cv_split: cross-validation split identifier
        :param is_standalone: True if model can be used independently from other models
        :param kwargs: other parameters
        """
        model = corner_cases(model)
        Wrapper.__init__(self, model, dataset_id, dataset_name, db_name, fit_params,
                         predict_params, score_params, model_id, cv_split, is_standalone)
        self.parameters = self.parse_parameters()

    def set_random_seed(self, seed, split_index, random_model, **kwargs):
        """
        Set model random state if possible.

        Parameters
        --------
        :param seed: random seed
        :param split_index: cross-validation split identifier
        :param random_model: whether the model should have the same random state for each cross-validation split
        :param kwargs: some parameters
        :return: None
        """
        if random_model:
            random_state = split_index
        else:
            random_state = seed
        try:
            self.model.set_params(**{"random_state": random_state})
        except (AttributeError, ValueError):
            pass
        self.cv_split = split_index
        self.parameters = self.parse_parameters()

    def parse_parameters(self) -> str:
        """
        Parse sklearn model parameters.

        Parameters
        --------
        :return: model parameters
        """
        return parse_sklearn_model(self.model)


class PipelineWrapper(Wrapper):
    """
    Class to wrap sklearn pipeline objects.
    """

    def __init__(self, model, dataset_id=None, dataset_name=None, db_name=None,
                 fit_params=None, predict_params=None, score_params=None, model_id=None,
                 cv_split=None, is_standalone=True, **kwargs):
        """
        Wrapper initialization.

        Parameters
        --------
        :param model: machine learning model
        :param dataset_id: data set identifier
        :param dataset_name: data set name
        :param db_name: database name
        :param fit_params: model's fit parameters
        :param predict_params: model's predict parameters
        :param score_params: model's score parameters
        :param model_id: model identifier
        :param cv_split: cross-validation split identifier
        :param is_standalone: True if model can be used independently from other models
        :param kwargs: other parameters
        """
        model = corner_cases(model)
        Wrapper.__init__(self, model, dataset_id, dataset_name, db_name, fit_params,
                         predict_params, score_params, model_id, cv_split, is_standalone)
        self.models = []
        self.models_id = []
        for step in model.steps:
            pipeline_step = SklearnWrapper(model=step[1], cv_split=cv_split, dataset_id=dataset_id,
                                           dataset_name=dataset_name, db_name=db_name, is_standalone=False)
            self.models.append(pipeline_step)
            self.models_id.append(pipeline_step.model_id)

        parameters = []
        for step in self.models:
            parameters.append(step.parameters)
        self.parameters = str(parameters)

    def save_model(self) -> None:
        """
        Save model into database.

        Parameters
        --------
        :return: None
        """
        self.models_id = []
        previous_step_id = -1
        # save each step separately
        for model in self.models:
            model.previous_step_id = previous_step_id
            model.save_model()
            self.models_id.append(model.model_id)
            previous_step_id = model.model_id
        # serialize model
        if not self.serialized_model:
            self.serialized_model = pickle.dumps(self.model, protocol=2)
        result = self.to_database()
        if result:
            model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
            self.model_id = model_id

    def load_model(self) -> Any:
        """
        Load model from database.

        Parameters
        --------
        :return: None
        """
        pipeline = []
        previous_step_id = -1
        i = 0
        is_fetchable = True
        # load each step separately (if present)
        for step in self.models:
            step.previous_step_id = previous_step_id
            if is_fetchable:
                fitted_step = step.load_model()
                if fitted_step.model_id and is_fetchable:
                    pipeline_step = ("id_" + str(fitted_step.model_id), fitted_step.model)
                    previous_step_id = fitted_step.model_id
                    self.models[i].is_fitted = True
                    self.models_id[i] = previous_step_id
                else:
                    # from now on the pipeline will be different, you can't fetch from database other models
                    is_fetchable = False
            if not is_fetchable:
                pipeline_step = ("n_" + str(i), copy.deepcopy(step.model))
            pipeline.append(pipeline_step)
            i += 1
        self.model.steps = pipeline
        result = self.from_database()
        if result:
            model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
            self.is_fitted = True
            self.model_id = model_id
            self.model = pickle.loads(serialized_model)
            self.serialized_model = serialized_model
        return self

    def set_random_seed(self, seed, split_index, random_model, **kwargs):
        """
        Set model random state if possible.

        Parameters
        --------
        :param seed: random seed
        :param split_index: cross-validation split identifier
        :param random_model: whether the model should have the same random state for each cross-validation split
        :param kwargs: some parameters
        :return: None
        """
        if random_model:
            random_state = split_index
        else:
            random_state = seed
        for parameter in list(self.model.get_params().keys()):
            if "random_state" in parameter:
                self.model.set_params(**{parameter: random_state})
        self.cv_split = split_index
        # set random seed of pipeline steps
        for model in self.models:
            model.set_random_seed(seed, split_index, random_model)

    def fit(self, x_train, y_train, **kwargs):
        """
        Fit model with some samples.

        Parameters
        --------
        :param x: train data
        :param y: train labels
        :param kwargs: some parameters
        :return: None
        """
        x_train_t = x_train
        i = 0
        for pipeline_step, model in zip(self.model.steps, self.models):
            if not model.is_fitted:
                # print("NOT FITTED!")
                pipeline_step[1].fit(x_train_t, y_train, )
                self.models[i] = SklearnWrapper(model=copy.deepcopy(pipeline_step[1]), cv_split=self.cv_split,
                                                dataset_id=self.dataset_id, dataset_name=self.dataset_name,
                                                db_name=self.db_name, is_standalone=False)
                self.models[i].is_fitted = True
            if hasattr(pipeline_step[1], "transform"):
                x_train_t = pipeline_step[1].transform(x_train_t)
            i += 1
        self.is_fitted = True


class KerasWrapper(Wrapper):
    """
    Class to wrap keras objects.
    """

    def __init__(self, model: Any, dataset_id=None, dataset_name=None, db_name=None,
                 fit_params=None, predict_params=None, score_params=None, model_id=None,
                 cv_split=None, is_standalone=True, **kwargs):
        """
        Wrapper initialization.

        Parameters
        --------
        :param model: machine learning model
        :param dataset_id: data set identifier
        :param dataset_name: data set name
        :param db_name: database name
        :param fit_params: model's fit parameters
        :param predict_params: model's predict parameters
        :param score_params: model's score parameters
        :param model_id: model identifier
        :param cv_split: cross-validation split identifier
        :param is_standalone: True if model can be used independently from other models
        :param kwargs: other parameters
        """
        Wrapper.__init__(self, model, dataset_id, dataset_name, db_name, fit_params,
                         predict_params, score_params, model_id, cv_split, is_standalone)
        self.parameters = self.parse_parameters()

    def load_model(self, **kwargs) -> Any:
        """
        Load model from database.

        Parameters
        --------
        :return: None
        """
        result = self.from_database()
        if result:
            model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
            self.is_fitted = True
            self.model_id = model_id
            self.model = load_neural_model(serialized_model)
            self.serialized_model = serialized_model
        else:
            self.is_fitted = False

    def save_model(self, **kwargs) -> None:
        """
        Save model into database.

        Parameters
        --------
        :return: None
        """
        if not self.serialized_model:
            self.serialized_model = save_neural_model(self.model)
        result = self.to_database()
        if result:
            model_id, model_type, model_class, serialized_model, fit_parameters, is_standalone = result
            self.model_id = model_id

    def set_random_seed(self, seed: int, split_index, random_model, **kwargs):
        """
        Set model random state if possible.

        Parameters
        --------
        :param seed: random seed
        :param split_index: cross-validation split identifier
        :param random_model: whether the model should have the same random state for each cross-validation split
        :param kwargs: some parameters
        :return: None
        """
        if random_model:
            random_state = split_index
        else:
            random_state = seed
        tf.set_random_seed(seed=random_state)
        reset_weights(self.model, random_state)
        self.cv_split = split_index
        self.parameters = self.parse_parameters()

    def parse_parameters(self) -> str:
        """
        Parse model parameters.

        Parameters
        --------
        :return: model parameters
        """
        return parse_neural_model(self.model)

    def fit(self, x, y, **kwargs):
        """
        Fit model with some samples.

        Parameters
        --------
        :param x: train data
        :param y: train labels
        :param kwargs: some parameters
        :return: None
        """
        if not self.is_fitted:
            fit_params = json.loads(self.fit_parameters)
            self.model.fit(x, y, **fit_params)
            self.is_fitted = True

    def predict(self, x, **kwargs) -> Any:
        """
        Predict labels for some input samples.

        Parameters
        --------
        :param x: input data
        :param kwargs: some parameters
        :return: predictions
        """
        predict_params = json.loads(self.predict_parameters)
        return self.model.predict(x, **predict_params)

    def score(self, x, y, **kwargs) -> Any:
        """
        Compute score for some input samples.

        Parameters
        --------
        :param x: input data
        :param y: input labels
        :param kwargs: some parameters
        :return: score
        """
        score_params = json.loads(self.score_parameters)
        return self.model.score(x, y, **score_params)


def parse_sklearn_model(model):
    """
    Parse sklearn model parameters.

    Parameters
    --------
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

    Parameters
    --------
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


def corner_cases(model: Any) -> Any:
    """
    Check parameter synonyms.

    Parameters
    --------
    :param model: sklearn model
    :return: model
    """
    if isinstance(model, RandomForestClassifier):
        if getattr(model, "n_estimators") == "warn":
            setattr(model, "n_estimators", 10)
    return model


def dict_to_json(dictionary: Union[dict, str]) -> str:
    """
    Sort dictionary by key and transform it into a string.

    Parameters
    --------
    :param dictionary: python dictionary
    :return: sorted dictionary as string
    """
    if isinstance(dictionary, str):
        dictionary = json.loads(dictionary)
    # sort dictionary by key
    if dictionary:
        sorted_dictionary = {}
        for param_key in sorted(dictionary):
            sorted_dictionary[param_key] = dictionary[param_key]
    else:
        sorted_dictionary = {}
    return json.dumps(sorted_dictionary)


def load_neural_model(model_bytes) -> Any:
    """
    Load keras model from binaries.

    Parameters
    --------
    :param model_bytes: serialized keras model
    :return: keras model
    """
    temp = os.path.join("./tmp", "temp.h5")
    with open(temp, 'wb') as output_file:
        output_file.write(model_bytes)
    return keras.models.load_model(temp)


def save_neural_model(model) -> Any:
    """
    Serialize keras model.

    Parameters
    --------
    :param model: keras model
    :return: serialized keras model
    """
    if not os.path.isdir("./tmp"):
        os.mkdir("./tmp")
    temp = os.path.join("./tmp", "temp.h5")
    model.save(temp)
    with open(temp, 'rb') as input_file:
        fitted_model = input_file.read()
    return fitted_model
