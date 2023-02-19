import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_precision_recall
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, \
    f1_score, roc_auc_score, mean_squared_error

from metrics.abstract_metric import AbstractMetric


class AccuracyMetric(AbstractMetric):
    name = "accuracy"
    threshold = 0.9
    description = "Accuracy calculates the proportion of correct predictions out of all the predictions made by the " \
                  "model "
    suggestion = "Try to use a more complex model or to add more data to the training set"

    def calculate(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()


class PrecisionMetric(AbstractMetric):
    name = "precision"
    threshold = 0.9
    description = "Precision measures how many observations predicted as positive are in fact positive"
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
    description = "recall calculates the proportion of true positive predictions out of all the actual positive instances"
    suggestion = "Try to adjust the threshold for classifying positive cases, to make the model more conservative or " \
                 "liberal"

    def calculate(self) -> float:
        return recall_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        pass


class F1Metric(AbstractMetric):
    name = "F1 score"
    threshold = 0.9
    description = "F1 score is an harmonic mean of precision and recall. It is commonly used when the dataset is " \
                  "imbalanced. "
    suggestion = "try to use oversampling or undersampling techniques to balance the dataset"

    def calculate(self) -> float:
        return f1_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        labels = ['Class 0', 'Class 1']
        plt.bar(labels, self.calculate())
        plt.title('F1 score per class')
        plt.xlabel('Class')
        plt.ylabel('F1 score')
        plt.show()


class AUCMetric(AbstractMetric):
    name = "AUC (Area Under the ROC Curve)"
    description = "AUC (Area Under the ROC Curve) calculates the area under the ROC curve, which plots the true " \
                  "positive rate against the false positive rate, the closer to 1, the better. A score of 0.5 is " \
                  "equivalent to random guessing. "
    suggestion = "try to use a different algorithm or to add more data to the training set"

    @property
    def threshold(self) -> float:
        def perf_measure(y_actual, y_hat):
            TP = 0
            FP = 0
            TN = 0
            FN = 0

            for i in range(len(y_hat)):
                if y_actual[i] == y_hat[i] == 1:
                    TP += 1
                if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                    FP += 1
                if y_actual[i] == y_hat[i] == 0:
                    TN += 1
                if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                    FN += 1

            return TP, FP, TN, FN

        TP, FP, TN, FN = perf_measure(self.y_true, self.y_pred)
        youdens_j = (TP / (TP + FN)) + (TN / (TN + FP)) - 1

        return youdens_j

    def is_perform_well(self) -> bool:
        return self.threshold > 0.5

    def calculate(self) -> float:
        return roc_auc_score(self.y_true, self.y_pred, multi_class='ovr')

    def suggestion_plot(self):
        fpr, tpr, _ = metrics.roc_curve(self.y_true, self.y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()


class MCCMetric(AbstractMetric):
    name = "Matthew's Correlation Coefficient (MCC)"
    threshold = 0.3
    description = "Matthew's Correlation Coefficient (MCC) measures the quality of a binary classification by taking " \
                  "into account true positives, true negatives, false positives, and false negatives and it is " \
                  "commonly used when the dataset is imbalanced. A high value for MCC (close to 1) indicates good " \
                  "performance, while a low value (close to 0 or below) indicates poor performance. "
    suggestion = "To improve performance, try adjusting the threshold of the classifier or consider modifying the " \
                 "feature set."

    def suggestion_plot(self):
        pass

    def calculate(self) -> float:
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        numerator = (tp * tn) - (fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return numerator / denominator


class MSEMetric(AbstractMetric):
    name = "Mean Squared Error (MSE)"
    description = "Mean Squared Error (MSE) is a popular regression metric which measures the average squared " \
                  "difference between the true and predicted values. "
    suggestion = "Lower the Mean Squared Error (MSE) value by tuning model hyperparameters or adjusting the training " \
                 "data."

    @property
    def threshold(self) -> float:
        var = np.var(self.y_true)
        return float(var)

    def suggestion_plot(self):
        pass

    def is_perform_well(self) -> bool:
        return self.calculate() < self.threshold

    def calculate(self) -> float:
        return mean_squared_error(self.y_true, self.y_pred)
