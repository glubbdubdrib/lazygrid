# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero and Giovanni Squillero
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
from keras.layers import Dense
from keras.initializers import glorot_uniform  # Or your initializer of choice
from keras import optimizers
import keras.backend as K


def reset_weights(model: Sequential, seed: int = 42):
    """
    Random reset of neural network's weights.

    Parameters
    --------
    :param model: Keras, theano, or tensorflow model
    :param seed: seed used to make results reproducible
    :return: None
    """

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.set_random_seed(seed)

    # get old weights
    initial_weights = model.get_weights()

    # check backend
    backend_name = K.backend()
    if backend_name == 'tensorflow':
        # # 5. Configure a new global `tensorflow` session
        # from keras import backend as K
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)
        k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
    elif backend_name == 'theano':
        k_eval = lambda placeholder: placeholder.eval()
    else:
        raise ValueError("Unsupported backend")

    # set new weights only if they are trainable
    new_weights = []
    for w, l in zip(initial_weights, model.layers):
        if l.trainable:
            new_weights.append(k_eval(glorot_uniform(seed=seed)(w.shape)))
        else:
            new_weights.append(w)
    model.set_weights(new_weights)


def keras_classifier(layers: list, input_shape: tuple, n_classes: int,
                     verbose: bool = True, metrics: list = ["accuracy"],
                     model_name: str = "MyKerasNet", lr: float = 0.1) -> Sequential:
    """
    Generate keras feed-forward neural model for classification.

    Parameters
    --------
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
