import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from typing import Dict, List, Any

class MetricsCollector:
    # Collects and aggregates evaluation metrics
    
    def compute_classification_metrics(self, y_true: Any, y_pred: Any) -> Dict:
        # Computes standard classification metrics (accuracy, F1, etc.)
        ...
    
    def compute_weak_supervision_metrics(self, weak_labels: Any, true_labels: Any) -> Dict:
        # Computes metrics specific to weak supervision quality
        ...
    
    def aggregate_results(self, results_list: List[Dict]) -> Dict:
        # Aggregates metrics across multiple runs or folds
        ...


class WMRCMetrics(MetricsCollector):
    # WMRC-specific metrics computation utilities
    
    def compute_classification_metrics(self, y_true: Any, y_pred: Any) -> Dict:
        # Computes accuracy, log loss, and Brier score for WMRC predictions
        y_true_squeezed = np.squeeze(y_true)
        
        metrics = {
            'accuracy': accuracy_score(y_true_squeezed, np.argmax(y_pred, axis=1)),
            'log_loss': log_loss(y_true_squeezed, y_pred),
            'brier_score': self.multi_brier(y_true_squeezed, y_pred),
            'error_rate': 1 - accuracy_score(y_true_squeezed, np.argmax(y_pred, axis=1))
        }
        return metrics
    
    def multi_brier(self, labels: Any, pred_probs: Any) -> float:
        # Computes multiclass Brier score for probabilistic predictions
        n_class = int(np.max(labels) + 1)
        labels_onehot = (np.arange(n_class) == labels[..., None]).astype(int)
        sq_diff = np.square(labels_onehot - pred_probs)
        return np.mean(np.sum(sq_diff, axis=1))
    
    def aggregate_results(self, results_list: List[Dict]) -> Dict:
        # Aggregates metrics across multiple experimental trials with mean/std
        aggregated = {}
        
        for key in results_list[0].keys():
            if isinstance(results_list[0][key], (int, float)):
                values = [result[key] for result in results_list]
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                
        return aggregated
    
    def get_result_filename(self, dataset_name: str, constraint_name: str, **kwargs) -> str:
        # Generates standardized result filename for WMRC experiments
        labeled_set = kwargs.get('labeled_set', 'valid')
        bound_method = kwargs.get('bound_method', 'binomial')
        use_inequality_consts = kwargs.get('use_inequality_consts', True)
        add_mv_const = kwargs.get('add_mv_const', False)
        n_labeled = kwargs.get('n_max_labeled')
        
        components = [
            "WMRC",
            dataset_name,
            constraint_name,
            self._get_paradigm_string(bound_method, labeled_set),
            self._get_labeled_count_string(n_labeled, use_inequality_consts, labeled_set),
            self._get_labeled_set_string(labeled_set),
            self._get_constraint_string(use_inequality_consts),
            self._get_bound_string(bound_method, labeled_set),
            self._get_mv_string(add_mv_const)
        ]
        
        return "_".join(filter(None, components)) + ".mat"
    
    def _get_paradigm_string(self, bound_method: str, labeled_set: str) -> str:
        # Helper to generate paradigm string for filename
        return 'unsup_' if bound_method == 'unsupervised' else 'semisup_'
    
    def _get_labeled_count_string(self, n_labeled: int, use_inequality_consts: bool, labeled_set: str) -> str:
        # Helper to generate labeled count string for filename
        if n_labeled and n_labeled > 0 and use_inequality_consts and labeled_set == 'valid':
            return str(n_labeled)
        return ''
    
    
