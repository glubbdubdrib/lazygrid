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

import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pycm


def plot_confusion_matrix(confusion_matrix: pd.DataFrame, font_scale: float,
                          file_name: str, title: str = 'Confusion matrix') -> None:
    """
    Generate confusion matrix figure.

    Parameters
    --------
    :param confusion_matrix: confusion matrix dataframe
    :param font_scale: font size
    :param file_name: figure file name
    :param title: figure title
    :return: None
    """

    sns.set(font_scale=font_scale)

    plt.figure()
    ax = sns.heatmap(confusion_matrix, annot=True, vmin=0, linewidths=.3, cmap="Greens", square= True)
    ax.set(xlabel='Prediction', ylabel='True', title=title)
    plt.savefig(file_name, dpi=800)
    plt.close()

    return


def one_hot_list_to_categorical(y_one_hot_list: List[np.ndarray]) -> np.ndarray:
    """
    Transform list of one-hot-encoded labels into a categorical array of labels.

    Parameters
    --------
    :param y_one_hot_list: one-hot-encoded list of labels
    :return: categorical array of labels
    """
    y_categorical_list = []
    for y_one_hot in y_one_hot_list:
        y_categorical_list.append(np.argmax(y_one_hot, axis=1))
    return np.hstack(y_categorical_list)


def generate_confusion_matrix(model_id: int, model_name: str,
                              y_pred_list: List[np.ndarray], y_true_list: List[np.ndarray],
                              class_names: dict = None, font_scale: float = 1,
                              output_dir: str = "./figures",
                              encoding: str = "categorical") -> pycm.ConfusionMatrix:
    """
    Generate and save confusion matrix.

    Parameters
    --------
    :param model_id: model identifier
    :param model_name: model name
    :param y_pred_list: predicted labels list
    :param y_true_list: true labels list
    :param class_names: dictionary of label names like {0: "Class 1", 1: "Class 2"}
    :param font_scale: figure font size
    :param output_dir: output directory
    :param encoding: kind of label encoding
    :return: confusion matrix object
    """

    # transform labels
    if encoding == "categorical":
        y_pred = np.hstack(y_pred_list)
        y_true = np.hstack(y_true_list)
    elif encoding == "one-hot":
        y_pred = one_hot_list_to_categorical(y_pred_list)
        y_true = one_hot_list_to_categorical(y_true_list)
    else:
        return None

    conf_mat = pycm.ConfusionMatrix(actual_vector=y_true, predict_vector=y_pred)

    # rename classes
    if class_names:
        conf_mat.relabel(mapping=class_names)

    conf_mat_pd = pd.DataFrame.from_dict(conf_mat.matrix)

    # figure title and file name
    name = model_name + "_" + str(model_id)
    title = model_name + " " + str(model_id)
    file_name = os.path.join(output_dir, "conf_mat_" + name + ".png")
    title = title
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    plot_confusion_matrix(conf_mat_pd, font_scale, file_name, title)

    return conf_mat


def plot_boxplots(score_list, labels, file_name, title, save = True):
    # box plots
    cv = np.stack(score_list, axis=1)
    plt.figure()
    results = plt.boxplot(cv, notch=True, labels=labels)
    if save: plt.savefig(file_name, dpi=800)
    plt.show()
    return results
