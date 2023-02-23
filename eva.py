from typing import List

from numpy import ndarray

from constants import Sensitivity
from metrics import AbstractMetric
from metrics import PrecisionMetric, RecallMetric, AccuracyMetric, F1Metric, AUCMetric, MCCMetric, MSEMetric, \
    BrierMetric


class Eva:
    def __init__(self, y_true: ndarray, y_pred: ndarray, y_prob=None, sensitivity: Sensitivity = Sensitivity.MEDIUM):
        precision_metric = PrecisionMetric(y_true, y_pred, sensitivity)
        recall_metric = RecallMetric(y_true, y_pred, sensitivity)
        accuracy_metric = AccuracyMetric(y_true, y_pred, sensitivity)
        f1_metric = F1Metric(y_true, y_pred, sensitivity)
        auc_metric = AUCMetric(y_true, y_pred, sensitivity)
        mcc_metric = MCCMetric(y_true, y_pred, sensitivity)
        mse_metric = MSEMetric(y_true, y_pred, sensitivity)
        brier_metric = BrierMetric(y_true, y_pred, sensitivity, y_prob)
        self.metrics: List[AbstractMetric] = [precision_metric, recall_metric, accuracy_metric, f1_metric, auc_metric,
                                              mcc_metric, mse_metric, brier_metric]

    def evaluate(self):
        for metric in self.metrics:
            if not metric.is_perform_well():
                print(f"The metric {metric.name} not perform well")
                print(metric.description)
                print(metric.suggestion)
                metric.suggestion_plot()
                print("=====================================================")
