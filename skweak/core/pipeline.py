from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class WeakPipeline(Pipeline, ABC):
    """Abstract base class for weak supervision pipelines"""

    def __init__(self, steps, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

    @abstractmethod
    def fit(self, X, L, y=None, **fit_params):
        """Fit pipeline with features X and weak labels L"""
        pass


class ExperimentPipeline(BaseEstimator, ABC):
    """Abstract base class for experiment pipelines"""

    def __init__(self):
        self.results_ = None
        self.best_params_ = None
        self.cv_results_ = None

    @abstractmethod
    def run_experiment(self, X, L, y, **experiment_params):
        """Run the experiment pipeline"""
        pass

    @abstractmethod
    def get_results(self):
        """Get experiment results"""
        pass
