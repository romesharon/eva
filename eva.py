from numpy import ndarray

from metrics import PrecisionMetric, RecallMetric, AccuracyMetric, ConfusionMetric, F1Metric


class Eva:
    def __init__(self, y_true: ndarray, y_pred: ndarray):
        generic_threshold = 0.6
        precision_metric = PrecisionMetric(y_true, y_pred)
        recall_metric = RecallMetric(y_true, y_pred)

        accuracy_metric = AccuracyMetric(y_true, y_pred)
        f1_metric = F1Metric(y_true, y_pred)
        confusion_metric = ConfusionMetric(y_true, y_pred)
        self.metrics = [precision_metric, recall_metric, confusion_metric]

    def evaluate(self):
        for metric in self.metrics:
            metric.calculate()
