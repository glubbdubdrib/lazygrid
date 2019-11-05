import sys
from typing import Callable
import json
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class PipelineWrapper(object):
    """
    Class to wrap sklearn pipeline objects.
    """

    def __init__(self, model, fit_params):

        assert isinstance(model, Pipeline)

        self.models = []
        self.models_id = []

        # wrap pipeline steps
        for step in model.steps:
            pipeline_step = ModelWrapper(step[1], fit_params, is_standalone=False)

            self.models.append(pipeline_step)
            self.models_id.append(pipeline_step.model_id)

        self.models_id = str(self.models_id)


def _corner_cases(model):
    if isinstance(model, RandomForestClassifier):
        if getattr(model, "n_estimators") == "warn":
            setattr(model, "n_estimators", 10)
    return model


def _parse_fit_params(fit_params):
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


class ModelWrapper(object):
    """
    Class to wrap model objects (sklearn, keras, tensorflow).
    """

    def __init__(self, model, fit_params=None, model_id=None, is_standalone=True):

        self.model_id = model_id
        self.model = self.fetch_model(model)
        self.model_type = str(model.__module__).split(".")[0]
        self.model_name = str(model.__class__.__name__).split(".")[0]
        self.version = sys.modules[self.model_type].__version__

        self.parameters = None
        self.fit_parameters = _parse_fit_params(fit_params)
        self.models = None
        self.models_id = None
        self.is_standalone = is_standalone

        if self.model_type == "sklearn":
            if isinstance(model, Pipeline):
                pipeline = PipelineWrapper(model, fit_params)
                self.models = pipeline.models
            else:
                self._parse_sklearn_model(model)
        elif self.model_type in ["keras", "tensorflow"]:
            self._parse_neural_model(model)
        else:
            raise ModuleNotFoundError

    def fetch_model(self, model=None):
        if self.model:
            return self.model
        if model:
            if hasattr(model, "_model"):
                return getattr("_model")
            else:
                return _corner_cases(model)
        else:
            return None

    def _parse_sklearn_model(self, model):
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

        self.parameters = ", ".join(args)

    def _parse_neural_model(self, model):
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

        self.parameters = str(parameters)
