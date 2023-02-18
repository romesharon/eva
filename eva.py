from numpy import ndarray

from metrics import PrecisionMetric, RecallMetric, FalsePositiveMetric, FalseDiscoveryMetric, \
    FalseNegativeMetric, TrueNegativeMetric, TruePositiveMetric, AccuracyMetric, ConfusionMetric, F1Metric, \
    NegativePredictiveMetric


class Eva:
    def __init__(self, y_true: ndarray, y_pred: ndarray):
        generic_threshold = 0.6
        precision_metric = PrecisionMetric(y_true, y_pred, generic_threshold)
        recall_metric = RecallMetric(y_true, y_pred, generic_threshold)
        false_positive_metric = FalsePositiveMetric(y_true, y_pred, generic_threshold)
        false_discovery_metric = FalseDiscoveryMetric(y_true, y_pred, generic_threshold)
        false_negative_metric = FalseNegativeMetric(y_true, y_pred, generic_threshold)
        true_negative_metric = TrueNegativeMetric(y_true, y_pred, generic_threshold)
        true_positive_metric = TruePositiveMetric(y_true, y_pred, generic_threshold)
        accuracy_metric = AccuracyMetric(y_true, y_pred, generic_threshold)
        f1_metric = F1Metric(y_true, y_pred, generic_threshold)
        negative_predictive_metric = NegativePredictiveMetric(y_true, y_pred, generic_threshold)
        confusion_metric = ConfusionMetric(y_true, y_pred, generic_threshold)
        self.metrics = [precision_metric, recall_metric, false_positive_metric, false_discovery_metric,
                        false_negative_metric, true_negative_metric, true_positive_metric, accuracy_metric,
                        f1_metric, negative_predictive_metric, confusion_metric]

    def evaluate(self):
        for metric in self.metrics:
            metric.calculate()
