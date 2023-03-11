import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, \
    f1_score, roc_auc_score, mean_squared_error, PrecisionRecallDisplay, brier_score_loss

from src.eva.constants import Sensitivity
from src.eva.metrics.abstract_metric import AbstractMetric


class AccuracyMetric(AbstractMetric):
    name = "Accuracy"
    threshold = {Sensitivity.LOW: 0.85, Sensitivity.MEDIUM: 0.9, Sensitivity.HIGH: 0.95}
    description = "Accuracy calculates the proportion of correct predictions out of all the predictions made by the " \
                  "model."
    suggestion = "Try to use a more complex model or to add more data to the training set, in addition, you can try " \
                 "to tune your hyperparametrs in order to get a better performance. "

    def calculate(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()


class PrecisionMetric(AbstractMetric):
    name = "Precision"
    threshold = {Sensitivity.LOW: 0.85, Sensitivity.MEDIUM: 0.9, Sensitivity.HIGH: 0.95}
    description = "Precision measures how many observations predicted as positive are in fact positive."
    suggestion = "Try to adjust the decision threshold of your model to improve the precision score of your binary " \
                 "classification task. Lowering the threshold can increase the number of true positives but may also " \
                 "increase false positives, while raising the threshold can decrease false positives but may also " \
                 "decrease true positives "

    def calculate(self) -> float:
        return precision_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        display = PrecisionRecallDisplay.from_predictions(self.y_true, self.y_pred)
        display.plot()

        _ = display.ax_.set_title("Binary classifcation Precision-Recall curve")


class RecallMetric(AbstractMetric):
    name = "Recall"
    threshold = {Sensitivity.LOW: 0.85, Sensitivity.MEDIUM: 0.9, Sensitivity.HIGH: 0.95}
    description = "Recall calculates the proportion of true positive predictions out of all the actual positive " \
                  "instances."
    suggestion = "Try to increase the size of your dataset or apply data augmentation techniques. Increasing the size " \
                 "of your dataset can provide more examples of the positive class, which can improve the model's " \
                 "ability to identify true positives. "

    def calculate(self) -> float:
        return recall_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        pass


class F1Metric(AbstractMetric):
    name = "F1 score"
    threshold = {Sensitivity.LOW: 0.85, Sensitivity.MEDIUM: 0.9, Sensitivity.HIGH: 0.95}
    description = "F1 score is an harmonic mean of precision and recall. It is commonly used when the dataset is " \
                  "imbalanced. "
    suggestion = "Try to balance precision and recall to improve F1 score by adjusting the decision threshold of your " \
                 "model or using algorithms that are specifically designed to optimize the F1 score. "

    def calculate(self) -> float:
        return f1_score(self.y_true, self.y_pred)

    def suggestion_plot(self):
        labels = ['Class 0', 'Class 1']
        plt.bar(labels,
                [f1_score(self.y_true, self.y_pred, pos_label=0), f1_score(self.y_true, self.y_pred, pos_label=1)])
        plt.title('F1 score per class')
        plt.xlabel('Class')
        plt.ylabel('F1 score')
        plt.show()


class AUCMetric(AbstractMetric):
    name = "AUC (Area Under the ROC Curve)"
    description = "AUC (Area Under the ROC Curve) calculates the area under the ROC curve, which plots the true " \
                  "positive rate against the false positive rate, the closer to 1, the better. A score of 0.5 is " \
                  "equivalent to random guessing. "
    threshold = {Sensitivity.LOW: 0.55, Sensitivity.MEDIUM: 0.5, Sensitivity.HIGH: 0.45}

    suggestion = "Try to use algorithms that directly optimize AUC to improve the AUC metric of your binary " \
                 "classification task. The ROC-AUC maximization algorithm is one such approach that can directly " \
                 "optimize AUC metric score. "

    @property
    def threshold_calculate(self) -> float:
        def perf_measure(y_actuals, y_hats):
            TP = 0
            FP = 0
            TN = 0
            FN = 0

            for y_actual, y_hat in zip(y_actuals, y_hats):
                if y_actual == y_hat == 1:
                    TP += 1
                if y_hat == 1 and y_actual != y_hat:
                    FP += 1
                if y_actual == y_hat == 0:
                    TN += 1
                if y_hat == 0 and y_actual != y_hat:
                    FN += 1

            return TP, FP, TN, FN

        TP, FP, TN, FN = perf_measure(self.y_true, self.y_pred)
        youdens_j = (TP / (TP + FN)) + (TN / (TN + FP)) - 1

        return youdens_j

    def is_perform_well(self) -> bool:
        return self.threshold_calculate > self.threshold.get(self.sensitivity)

    def calculate(self) -> float:
        try:
            return roc_auc_score(self.y_true, self.y_pred, multi_class='ovr')
        except:
            pass

    def suggestion_plot(self):
        fpr, tpr, _ = metrics.roc_curve(self.y_true, self.y_prob)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()


class MCCMetric(AbstractMetric):
    name = "Matthew's Correlation Coefficient (MCC)"
    threshold = {Sensitivity.LOW: 0.3, Sensitivity.MEDIUM: 0.35, Sensitivity.HIGH: 0.4}
    description = "Matthew's Correlation Coefficient (MCC) measures the quality of a binary classification by taking " \
                  "into account true positives, true negatives, false positives, and false negatives and it is " \
                  "commonly used when the dataset is imbalanced. A high value for MCC (close to 1) indicates good " \
                  "performance, while a low value (close to 0 or below) indicates poor performance. "
    suggestion = "Try to use more complex models like neural networks or gradient boosting to improve the Mean " \
                 "Squared Error (MSE) metric. These models can capture complex nonlinear relationships in the data " \
                 "and improve predictive accuracy. "

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
    suggestion = "Try to use more complex models like neural networks or gradient boosting to improve the Mean " \
                 "Squared Error (MSE) metric. These models can capture complex nonlinear relationships in the data " \
                 "and improve predictive accuracy. "

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


class BrierMetric(AbstractMetric):
    name = "Brier Score"
    description = "Brier score is used to check the goodness of a predicted probability " \
                  "score. This is very similar to the mean squared error, but only applied for prediction probability " \
                  "scores, whose values range between 0 and 1. "
    suggestion = "Try to adjust the model's hyperparameters, such as the regularization strength, or to use a " \
                 "different algorithm that is better suited to the data, in addtion, try to increase the amount of " \
                 "data or improve the quality of the input features may also help to improve the model's performance. "
    threshold = {Sensitivity.LOW: 0.18, Sensitivity.MEDIUM: 0.15, Sensitivity.HIGH: 0.1}

    def suggestion_plot(self):
        pass

    def is_perform_well(self) -> bool:
        return self.calculate() < self.threshold.get(self.sensitivity)

    def calculate(self) -> float:
        return brier_score_loss(self.y_true, self.y_prob)
