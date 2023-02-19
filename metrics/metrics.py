import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, \
    f1_score

from metrics.abstract_metric import AbstractMetric
from scikitplot.metrics import plot_precision_recall


class AccuracyMetric(AbstractMetric):
    name = "accuracy"
    threshold = 0.9
    description = "calculates the proportion of correct predictions out of all the predictions made by the model"
    suggestion = "Try to use a more complex model or to add more data to the training set"

    def calculate(self):
        print("accuracy rate:", accuracy_score(self.y_true, self.y_pred))

    def suggestion_plot(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()


class PrecisionMetric(AbstractMetric):
    name = "precision"
    threshold = 0.9
    description = "measures how many observations predicted as positive are in fact positive"
    suggestion = "Try to adjust the threshold for classifying positive cases, to make the model more conservative or " \
                 "liberal"

    def calculate(self) -> float:
        return precision_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        fig, ax = plt.subplots()
        plot_precision_recall(self.y_true, self.y_pred, ax=ax)


class RecallMetric(AbstractMetric):
    name = "recall"
    threshold = 0.9
    description = "Calculates the proportion of true positive predictions out of all the actual positive instances"
    suggestion = "Try to adjust the threshold for classifying positive cases, to make the model more conservative or " \
                 "liberal "

    def calculate(self) -> float:
        return recall_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        fig, ax = plt.subplots()
        plot_precision_recall(self.y_true, self.y_pred, ax=ax)


class F1Metric(AbstractMetric):
    def calculate(self):
        print("f1 score:", f1_score(self.y_true, self.y_pred))


# class FalsePositiveMetric(AbstractMetric):
#     name= "false positive rate"
#     def calculate(self) -> float:
#         tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
#         print("false positive rate:", fp / (fp + tn))
#

# class FalseNegativeMetric(AbstractMetric):
#     def calculate(self):
#         tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
#         print("false negative rate:", fn / (tp + fn))
#

# class TrueNegativeMetric(AbstractMetric):
#     def calculate(self):
#         tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
#         print("true negative rate:", tn / (tn + fp))
#
#
# class NegativePredictiveMetric(AbstractMetric):
#     def calculate(self):
#         tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
#         print("negative predictive value:", tn / (tn + fn))


# class FalseDiscoveryMetric(AbstractMetric):
#     def calculate(self):
#         tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
#         print("false discovery rate:", fp / (tp + fp))
#
#
# class TruePositiveMetric(AbstractMetric):
#     def calculate(self):
#         tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
#         print("true positive rate:", tp / (tp + fn))
#


class ConfusionMetric(AbstractMetric):
    def calculate(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
