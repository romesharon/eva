from sklearn.metrics import classification_report


def evaluate_model(y_actual, y_pred):
    # Calculate accuracy
    accuracy = sum(y_actual == y_pred) / len(y_actual)
    print('Accuracy:', accuracy)

    # Generate classification report
    print('Classification Report:')
    report = classification_report(y_actual, y_pred)
    print(report)

    # Extract metrics for each class
    metrics = report.split('\n')[2:-3]
    metrics = [m.split() for m in metrics]
    metrics = [(m[0], float(m[1]), float(m[2]), float(m[3]), float(m[4])) for m in metrics]

    # Calculate dynamic thresholds based on class distribution and metric variance
    class_distribution = {c: sum(y_actual == c) for c in set(y_actual)}
    total_samples = len(y_actual)

    precision_thresholds = {}
    recall_thresholds = {}

    for c in class_distribution:
        class_samples = y_actual == c
        class_precision = metrics[c][2]
        class_recall = metrics[c][3]

        # Calculate precision threshold based on variance of precision in the class
        class_precision_var = class_precision * (1 - class_precision) / sum(class_samples)
        precision_threshold = max(0.2, class_precision - 2 * class_precision_var)
        precision_thresholds[c] = precision_threshold

        # Calculate recall threshold based on class distribution and variance of recall in the class
        class_recall_var = class_recall * (1 - class_recall) / sum(class_samples)
        recall_threshold = max(0.2, (class_distribution[c] / total_samples) * class_recall - 2 * class_recall_var)
        recall_thresholds[c] = recall_threshold

    # Identify poorly performing classes
    poorly_performing_classes = []
    for metric in metrics:
        c = metric[0]
        precision = metric[2]
        recall = metric[3]

        # Check if precision or recall is below the threshold
        if precision < precision_thresholds[c] or recall < recall_thresholds[c]:
            poorly_performing_classes.append(c)

    # Print summary of poorly performing classes
    if poorly_performing_classes:
        print('The model performs poorly on the following classes:', poorly_performing_classes)
    else:
        print('The model performs well on all classes.')
