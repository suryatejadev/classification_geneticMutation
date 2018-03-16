"""
The :mod:`scikitplot.metrics` module includes plots for machine learning
evaluation metrics e.g. confusion matrix, silhouette scores, etc.
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.calibration import calibration_curve

from scipy import interp

from scikitplot.helpers import binary_ks_curve, validate_labels
from scikitplot.helpers import cumulative_gain_curve


def plot_confmatrix(y_true, y_pred, labels=None, true_labels=None,
                          pred_labels=None, title=None, normalize=False,
                          hide_zeros=False, x_tick_rotation=0, ax=None,
                          figsize=None, cmap='Blues', title_fontsize="large",
                          text_fontsize="medium"):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        classes = unique_labels(y_true, y_pred)
    else:
        classes = np.asarray(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    if true_labels is None:
        true_classes = classes
    else:
        validate_labels(classes, true_labels, "true_labels")

        true_label_indexes = np.in1d(classes, true_labels)

        true_classes = classes[true_label_indexes]
        cm = cm[true_label_indexes]

    if pred_labels is None:
        pred_classes = classes
    else:
        validate_labels(classes, pred_labels, "pred_labels")

        pred_label_indexes = np.in1d(classes, pred_labels)

        pred_classes = classes[pred_label_indexes]
        cm = cm[:, pred_label_indexes]

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    elif normalize:
        ax.set_title('Normalized Confusion Matrix', fontsize=title_fontsize)
    else:
        ax.set_title('Confusion Matrix', fontsize=title_fontsize)

    image = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    plt.colorbar(mappable=image)
    x_tick_marks = np.arange(len(pred_classes))
    y_tick_marks = np.arange(len(true_classes))
    ax.set_xticks(x_tick_marks)
    ax.set_xticklabels(pred_classes, fontsize=text_fontsize,
                       rotation=x_tick_rotation)
    ax.set_yticks(y_tick_marks)
    ax.set_yticklabels(true_classes, fontsize=text_fontsize)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if not (hide_zeros and cm[i, j] == 0):
            ax.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=text_fontsize,
                    color="white" if cm[i, j] > thresh else "black")

    #ax.set_ylabel('True label', fontsize=text_fontsize)
    #ax.set_xlabel('Predicted label', fontsize=text_fontsize)
    ax.grid('off')

    return ax


