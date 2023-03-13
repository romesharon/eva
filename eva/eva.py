from typing import Dict

from numpy import ndarray
from scipy.stats import ttest_ind

from eva.constants import Sensitivity
from eva.metrics import PrecisionMetric, RecallMetric, AccuracyMetric, F1Metric, AUCMetric, MCCMetric, MSEMetric, \
    BrierMetric


class Eva:
    def __init__(self, y_true: ndarray, y_pred: ndarray,
                 y_prob=None, sensitivity: Sensitivity = Sensitivity.MEDIUM):
        self.y_pred_test = y_pred
        self.metrics = {
            "accuracy_metric": AccuracyMetric(y_true, y_pred, y_prob, sensitivity),
            "precision_metric": PrecisionMetric(y_true, y_pred, y_prob, sensitivity),
            "recall_metric": RecallMetric(y_true, y_pred, y_prob, sensitivity),
            "f1_metric": F1Metric(y_true, y_pred, y_prob, sensitivity),
            "mcc_metric": MCCMetric(y_true, y_pred, y_prob, sensitivity),
            "mse_metric": MSEMetric(y_true, y_pred, y_prob, sensitivity),
            "brier_metric": BrierMetric(y_true, y_pred, y_prob, sensitivity),
            "auc_metric": AUCMetric(y_true, y_pred, y_prob, sensitivity)
        }

    def evaluate(self) -> Dict[str, float]:
        metrics_score = {}
        for metric in self.metrics.values():
            value = metric.calculate()
            metrics_score[metric.name] = value
            print(f"Meric Name:{metric.name}, value: {value}")
            if not metric.is_perform_well():
                print(f"The metric {metric.name} not perform well")
                print(metric.description)
                print(metric.suggestion)
                metric.suggestion_plot()
            print("=====================================================")
        return metrics_score

    def check_overfitting(self, metrics=['accuracy'], tol=0.05, alpha=0.05, summary=True):
        """
            Check for overfitting by comparing evaluation metrics between training and testing sets, and
            testing for statistical significance of the difference using a two-tailed t-test, and/or checking
            whether the performance difference is within a tolerance level.
            Returns a tuple containing a boolean indicating whether there is overfitting, and a dictionary
            of evaluation metrics for the training and testing sets.
            """
        overfitting = False
        for metric_name in metrics:
            train_metric = self.train_metrics[metric_name].calculate()
            test_metric = self.test_metrics[metric_name].calculate()
            delta = train_metric - test_metric
            if delta > tol:
                t, p = ttest_ind(self.y_pred_train, self.y_pred_test)
                if p < alpha:
                    overfitting = True
                    print(f'Warning: Potential overfitting detected for metric {metric_name}.')
                    print(f'Training set performance: {train_metric}')
                    print(f'Testing set performance: {test_metric}')
                    print(f'Statistically significant difference in {metric_name} (p = {p}).')
            if summary:
                print(f'{metric_name.capitalize()}: Train {train_metric:.4f}, Test {test_metric:.4f}')
        return overfitting
