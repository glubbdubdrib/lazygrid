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
import seaborn as sns
import pandas as pd


# this function plots a confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names, file_name, title='Confusion matrix', save = True):
    """
    This function prints and plots the confusion matrix.
    """

    sns.set(font_scale=5/(2+len(class_names)))

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    plt.figure()
    ax = sns.heatmap(df_cm, annot=True, vmin=0, linewidths=.3, cmap="Greens", square= True)
    ax.set(xlabel='True', ylabel='Prediction', title=title)
    plt.savefig(file_name, dpi=800)
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
