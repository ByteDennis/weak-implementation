from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class WeakMetric(ABC):
    """Abstract base class for weak supervision metrics"""

    @abstractmethod
    def __call__(self, y_true, y_pred, **kwargs):
        """Compute the metric"""
        pass


class WeakScorer(ABC):
    """Abstract base class for weak supervision scorers"""

    @abstractmethod
    def score(self, estimator, X, y, **kwargs):
        """Score the estimator"""
        pass
