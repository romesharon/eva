import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, \
    f1_score


def precision_metric(y_true: ndarray, y_pred: ndarray):
    print("precision:", precision_score(y_true, y_pred))


def recall_metric(y_true: ndarray, y_pred: ndarray):
    print("recall:", recall_score(y_true, y_pred))


def false_positive_rate(y_true: ndarray, y_pred: ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("false positive rate:", fp / (fp + tn))


def false_negative_rate(y_true: ndarray, y_pred: ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("false negative rate:", fn / (tp + fn))


def true_negative_rate(y_true: ndarray, y_pred: ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("true negative rate:", tn / (tn + fp))


def negative_predictive_value(y_true: ndarray, y_pred: ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("negative predictive value:", tn / (tn + fn))


def false_discovery_rate(y_true: ndarray, y_pred: ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("false discovery rate:", fp / (tp + fp))


def true_positive_rate(y_true: ndarray, y_pred: ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("true positive rate:", tp / (tp + fn))


def accuracy(y_true: ndarray, y_pred: ndarray):
    print("true positive rate:", accuracy_score(y_true, y_pred))


def f1(y_true: ndarray, y_pred: ndarray):
    print("f1 score:", f1_score(y_true, y_pred))


def confusion_metric(y_true: ndarray, y_pred: ndarray):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
