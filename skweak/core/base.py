from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import accuracy_score
import numpy as np


class WeakEstimator(BaseEstimator, ABC):
    """Abstract base class for all weak supervision estimators"""

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the estimator to data"""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions on data"""
        pass

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels"""
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class WeakTransformer(WeakEstimator, TransformerMixin, ABC):
    """Abstract base class for weak supervision transformers"""

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the transformer to data"""
        pass

    @abstractmethod
    def transform(self, X):
        """Transform the data"""
        pass

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it"""
        return self.fit(X, y, **fit_params).transform(X)
    
class WeakClassifier(WeakEstimator, ClassifierMixin, ABC):
    """Abstract base class for weak supervision classifiers"""

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the classifier to data"""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict class labels for samples in X"""
        pass

    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        raise NotImplementedError("predict_proba not implemented")

    def decision_function(self, X):
        """Predict confidence scores for samples"""
        raise NotImplementedError("decision_function not implemented")


class LabelModel(WeakTransformer, ABC):
    """Abstract base class for label models"""

    def __init__(self):
        self.classes_ = None
        self.n_sources_ = None
        self.n_classes_ = None

    @abstractmethod
    def fit(self, L, y=None, **fit_params):
        """Fit label model on weak label matrix L"""
        pass

    @abstractmethod
    def transform(self, L):
        """Transform weak labels to aggregated labels"""
        pass

    def predict(self, L):
        """Predict hard labels from weak labels"""
        probs = self.transform(L)
        return np.argmax(probs, axis=1)

    def predict_proba(self, L):
        """Predict soft labels from weak labels"""
        return self.transform(L)


class EndModel(WeakClassifier, ABC):
    """Abstract base class for end models"""

    def __init__(self):
        self.classes_ = None
        self.feature_dim_ = None

    @abstractmethod
    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit end model on features X and labels y"""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict labels for features X"""
        pass


class JointModel(WeakClassifier, ABC):
    """Abstract base class for joint learning models"""

    def __init__(self):
        self.classes_ = None
        self.label_model_ = None
        self.end_model_ = None

    @abstractmethod
    def fit(self, X, L, y_val=None, **fit_params):
        """Jointly train on features X and weak labels L"""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict labels for features X"""
        pass
