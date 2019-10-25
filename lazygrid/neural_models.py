# -*- coding: utf-8 -*-
#
# Copyright 2019 - Barbiero Pietro and Squillero Giovanni
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

from keras import Sequential
from keras.layers import Dense, Activation, InputLayer
from sklearn.preprocessing import LabelEncoder
from tensorflow import set_random_seed
from keras.utils import to_categorical
from keras.initializers import glorot_uniform  # Or your initializer of choice
from keras import optimizers
import keras.backend as K


def reset_weights(model: Sequential, seed: int = 42):
    """
    Random reset of neural network's weights.

    :param model: Keras, theano, or tensorflow model
    :param seed: seed used to make results reproducible
    :return: None
    """

    # get old weights
    initial_weights = model.get_weights()

    # check backend
    backend_name = K.backend()
    if backend_name == 'tensorflow':
        k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
    elif backend_name == 'theano':
        k_eval = lambda placeholder: placeholder.eval()
    else:
        raise ValueError("Unsupported backend")

    # set new weights
    new_weights = [k_eval(glorot_uniform(seed=seed)(w.shape)) for w in initial_weights]
    model.set_weights(new_weights)


def keras_classifier(layers: list, input_shape: tuple, n_classes: int,
                     verbose: bool = True, metrics: list = ["accuracy"],
                     model_name: str = "MyKerasNet", lr: float = 0.1) -> Sequential:
    """
    Generate keras feed-forward neural model for classification.

    :param layers: list of layers
    :param input_shape: shape of input data
    :param n_classes: number of classes
    :param verbose: if True it prints the summary of the model
    :param metrics: metrics used by the optimizer
    :param model_name: name of the keras network
    :param lr: learning rate
    :return: compiled keras model
    """

    assert n_classes >= 2

    output_size = 1 if n_classes == 2 else n_classes
    if n_classes > 2:
        activation = "softmax"
        loss = "categorical_crossentropy"
    else:
        activation = "sigmoid"
        loss = "binary_crossentropy"

    model = Sequential()
    model.add(Dense(layers[0], activation="relu", input_shape=input_shape))
    for layer in layers[1:]:
        model.add(Dense(layer, activation="relu"))
    model.add(Dense(output_size, activation=activation))

    model.__name__ = model_name

    if verbose:
        model.summary()

    model.compile(loss=loss,
                  optimizer=optimizers.Adam(lr=lr),
                  metrics=metrics)

    return model