from metrics.metrics import precision_metric, recall_metric, false_positive_rate, confusion_metric, false_negative_rate, \
    true_negative_rate, accuracy, negative_predictive_value, false_discovery_rate, true_positive_rate, f1

metrics_functions = [precision_metric, recall_metric, false_positive_rate, false_negative_rate, true_negative_rate,
                     negative_predictive_value, false_discovery_rate, true_positive_rate, accuracy, f1,
                     confusion_metric]
