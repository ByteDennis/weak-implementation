import numpy as np
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from ..core.pipeline import ExperimentPipeline
from ..utils.sk_utils import combine_weak_and_labeled_data

class SemiSupervisedExperiment(ExperimentPipeline):
    """Semi-supervised learning experiment with weak supervision"""
    
    def __init__(self, ssl_method='label_spreading', weak_ratio=0.8):
        super().__init__()
        self.ssl_method = ssl_method
        self.weak_ratio = weak_ratio
        self.model_ = None
    
    def run_experiment(self, X, L, y, X_labeled=None, y_labeled=None, **experiment_params):
        """Run semi-supervised experiment"""
        
        # Combine weak and labeled data
        if X_labeled is not None:
            X_combined, y_combined = combine_weak_and_labeled_data(
                X, L, X_labeled, y_labeled
            )
        else:
            # Use weak supervision only
            X_combined, y_combined = X, L
        
        # Create semi-supervised model
        self.model_ = self._get_ssl_model()
        
        # Train model
        self.model_.fit(X_combined, y_combined)
        
        # Evaluate
        y_pred = self.model_.predict(X)
        accuracy = np.mean(y_pred == y)
        
        self.results_ = {
            'accuracy': accuracy,
            'ssl_method': self.ssl_method,
            'weak_ratio': self.weak_ratio
        }
        
        return self.results_
    
    def _get_ssl_model(self):
        """Get semi-supervised model"""
        models = {
            'label_spreading': LabelSpreading(),
            'label_propagation': LabelPropagation()
        }
        return models[self.ssl_method]
    
    def get_results(self):
        return self.results_
    
    
__all__ = ['SemiSupervisedExperiment']