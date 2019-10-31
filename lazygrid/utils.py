import numpy as np
from keras import Sequential, Model
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
import tensorflow as tf
from .neural_models import reset_weights
from .wrapper import ModelWrapper


def is_fitted(model: ModelWrapper, x: np.ndarray) -> bool:
    """
    Check if the pipeline step is fitted.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.feature_selection import SelectKBest, f_classif
    >>> from sklearn.datasets import make_classification
    >>> import lazygrid as lg
    >>>
    >>> x, y = make_classification()
    >>>
    >>> fs = SelectKBest(f_classif, k=5)
    >>> clf = RandomForestClassifier()
    >>> model = lg.ModelWrapper(Pipeline([('feature_selector', fs), ('clf', clf)]))
    >>>
    >>> is_fitted(model, x)
    False
    >>>
    >>> _ = model.model.fit(x, y)
    >>>
    >>> is_fitted(model, x)
    True

    Parameters
    --------
    :param step: pipeline step
    :param x: test data
    :return: True if step is fitted, False otherwise
    """
    x = x[:2]
    try:

        # keras models are fitted if they have been loaded
        if model.model_type in ["keras", "tensorflow"]:
            if model.model_id:
                return True
            else:
                return False

        # a sklearn model is fitted if it can predict or transform inpu data
        if hasattr(model.model, "transform"):
            model.model.transform(x)
        elif hasattr(model.model, "predict"):
            model.model.predict(x)
        else:
            return False

    except NotFittedError:
        return False

    return True


def set_random_seed(learner: ModelWrapper,
                     random_model: bool, split_index: int, seed: int) -> ModelWrapper:
    """
    Set model random seed for the sake of reproducibility.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> model = RandomForestClassifier()
    >>> seed = 42
    >>>
    >>> learner = set_random_seed(model, random_model=True, split_index=0, seed=seed)
    >>> learner.random_state
    0
    >>> learner = set_random_seed(model, random_model=False, split_index=0, seed=seed)
    >>> learner.random_state
    42

    Parameters
    --------
    :param learner: machine learning model
    :param random_model: True to set random state equal to `seed`; False to set random state equal to `split_index`
    :param split_index: cross-validation split identifier
    :param seed: random seed
    :return: True if step is fitted, False otherwise
    """

    if str(learner.__module__).split(".")[0] in ["keras", "tensorflow"]:

        # reset model weights if needed
        if random_model:
            tf.set_random_seed(seed=split_index)
            reset_weights(learner.model, split_index)
        else:
            tf.set_random_seed(seed=seed)
            reset_weights(learner.model, seed)

    elif isinstance(learner, Pipeline):

        # reset learner initialization if needed
        for parameter in list(learner.get_params().keys()):
            if "random_state" in parameter:
                if random_model:
                    learner.set_params(**{parameter: split_index})
                else:
                    learner.set_params(**{parameter: seed})

    # Learning with other models
    elif hasattr(learner, "fit") and hasattr(learner, "predict"):

        # reset model initialization if needed
        if hasattr(learner, "random_state"):
            if random_model:
                learner.set_params(**{"random_state": split_index})
            else:
                learner.set_params(**{"random_state": seed})

    return learner
