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

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'normal',
        'size': 14}
matplotlib.rc('font', **font)


# this function plots a confusion matrix
def plot_confusion_matrix(confusion_matrix, classes, fileName, title='Confusion matrix', save = True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmNormalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # attempt at creating a more meaningful visualization: values that are in the "wrong box"
    # (for which predicted label is different than true label) are turned negative
    for i in range(0, cmNormalized.shape[0]):
        for j in range(0, cmNormalized.shape[1]):
            if i != j: cmNormalized[i, j] *= -1.0

    fig = plt.figure()
    plt.imshow(cmNormalized, vmin=0.0, vmax=1.0, interpolation='nearest', cmap='Greens')  # cmap='RdYlGn')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cmNormalized.max() / 2.
    for i, j in itertools.product(range(cmNormalized.shape[0]), range(cmNormalized.shape[1])):
        text = "%.2f\n(%d)" % (abs(cmNormalized[i, j]), confusion_matrix[i, j])
        plt.text(j, i, text, horizontalalignment="center",
                 color="white" if cmNormalized[i, j] > thresh or cmNormalized[i, j] < -thresh else "black")

    plt.tight_layout()
    plt.ylabel('Prediction')
    plt.xlabel('True')

    fig.subplots_adjust(bottom=0.2)
    if save: plt.savefig(fileName, dpi=800)
    plt.close()

    return


def plot_boxplots(score_list, labels, file_name, title, save = True):
    # box plots
    cv = np.stack(score_list, axis=1)
    plt.figure()
    results = plt.boxplot(cv, notch=True, labels=labels)
    if save: plt.savefig(file_name, dpi=800)
    plt.show()
    return results
