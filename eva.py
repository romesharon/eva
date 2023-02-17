from metrics import metrics_functions


def evaluate(y_true, y_pred):
    for metric_function in metrics_functions:
        metric_function(y_true, y_pred)
