import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from ..core.metrics import WeakMetric, WeakScorer


class LabelModelMetric(WeakMetric):
    """Metrics specific to label model evaluation"""
    
    def __init__(self, coverage_penalty=0.0):
        self.coverage_penalty = coverage_penalty
    
    def __call__(self, L_true, L_pred, **kwargs):
        """Evaluate label model performance"""
        
        # Basic accuracy
        mask = L_true != -1  # Non-abstain positions
        if np.any(mask):
            accuracy = accuracy_score(L_true[mask], L_pred[mask])
        else:
            accuracy = 0.0
        
        # Coverage penalty
        coverage = np.mean(L_pred != -1)
        penalized_score = accuracy - self.coverage_penalty * (1 - coverage)
        
        return {
            'accuracy': accuracy,
            'coverage': coverage,
            'penalized_score': penalized_score
        }

class Scorer(WeakScorer):
    """Scorer for weak supervision pipelines"""
    
    def __init__(self, metric='accuracy', use_soft_labels=False):
        self.metric = metric
        self.use_soft_labels = use_soft_labels
        
        self.metric_funcs = {
            'accuracy': accuracy_score,
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score,
            'auc': roc_auc_score
        }
    
    def score(self, estimator, X, y, **kwargs):
        """Score the estimator"""
        
        if self.use_soft_labels and hasattr(estimator, 'predict_proba'):
            y_pred = estimator.predict_proba(X)
            if self.metric == 'auc':
                return roc_auc_score(y, y_pred[:, 1])
            else:
                y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = estimator.predict(X)
        
        metric_func = self.metric_funcs[self.metric]
        
        if self.metric in ['f1', 'precision', 'recall'] and len(np.unique(y)) > 2:
            return metric_func(y, y_pred, average='weighted')
        else:
            return metric_func(y, y_pred)

def evaluate_model(estimator, X_test, L_test, y_test, metrics=['accuracy', 'f1']):
    """Comprehensive evaluation of weak supervision models"""
    
    results = {}
    
    for metric in metrics:
        scorer = Scorer(metric=metric)
        score = scorer.score(estimator, X_test, y_test)
        results[metric] = score
    
    # Additional weak supervision specific metrics
    if hasattr(estimator, 'predict_proba'):
        y_proba = estimator.predict_proba(X_test)
        results['entropy'] = np.mean(-np.sum(y_proba * np.log(y_proba + 1e-8), axis=1))
    
    return results

__all__ = [
    'evaluate_model'
]