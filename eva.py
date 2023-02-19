from typing import List

from numpy import ndarray

from metrics import PrecisionMetric, RecallMetric, AccuracyMetric, F1Metric, AUCMetric, MCCMetric, MSEMetric
from metrics import AbstractMetric


class Eva:
    def __init__(self, y_true: ndarray, y_pred: ndarray):
        precision_metric = PrecisionMetric(y_true, y_pred)
        recall_metric = RecallMetric(y_true, y_pred)
        accuracy_metric = AccuracyMetric(y_true, y_pred)
        f1_metric = F1Metric(y_true, y_pred)
        auc_metric = AUCMetric(y_true, y_pred)
        mcc_metric = MCCMetric(y_true, y_pred)
        mse_metric = MSEMetric(y_true, y_pred)
        self.metrics: List[AbstractMetric] = [precision_metric, recall_metric, accuracy_metric, f1_metric, auc_metric,
                                              mcc_metric, mse_metric]

    def evaluate(self):
        for metric in self.metrics:
            if not metric.is_perform_well():
                print(f"The metric {metric.name} not perform well")
                print(metric.description)
                print(metric.suggestion)
                metric.suggestion_plot()
                print("=====================================================")
