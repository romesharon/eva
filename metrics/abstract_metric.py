from abc import ABC, abstractmethod

from numpy import ndarray


class AbstractMetric(ABC):
    def __init__(self, y_true: ndarray, y_pred: ndarray):
        self.y_true = y_true
        self.y_pred = y_pred

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def threshold(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass

    @property
    @abstractmethod
    def suggestion(self):
        pass

    @abstractmethod
    def suggestion_plot(self):
        pass

    @abstractmethod
    def calculate(self) -> float:
        pass

    def is_perform_well(self) -> bool:
        return self.calculate() > self.threshold
