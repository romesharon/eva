from abc import ABC, abstractmethod

from numpy import ndarray


class AbstractMetric(ABC):
    def __init__(self, y_true: ndarray, y_pred: ndarray, threshold: float):
        self.y_true = y_true
        self.y_pred = y_pred
        self.threshold = threshold

    @abstractmethod
    def calculate(self):
        pass
