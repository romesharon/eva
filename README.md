Eva - Automatic Evaluation Metrics Analysis and Explanation for Binary Classification
=====================================================================================

Eva is a Python package that provides automatic analysis and explanation of evaluation metrics for binary classification models. It allows users to easily evaluate the performance of their models using standard metrics such as accuracy, precision, recall, F1 score, and ROC AUC, and provides detailed explanations of these metrics to help users understand how their model is performing. Additionally, Eva identifies the metrics that perform poorly and provides suggestions to fix them.

Installation
------------

To install Eva, you can use pip:

`pip install eva`

Usage
-----

Here's an example of how to use Eva to evaluate the performance of a binary classification model:


`from eva import Eva

# Example y_true, y_pred, and y_prob
y_true = [0, 1, 1, 0, 1, 0, 1, 0]
y_pred = [0, 0, 1, 1, 1, 0, 0, 0]
y_prob = [0.1, 0.4, 0.6, 0.7, 0.8, 0.3, 0.2, 0.5]

# Create an Eva object and evaluate the model
eva = Eva(y_true, y_pred, y_prob)
eva.evaluate()
