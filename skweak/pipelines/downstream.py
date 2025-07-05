import numpy as np
from sklearn.pipeline import Pipeline
from ..core import ExperimentPipeline, LabelModel, EndModel
from ..utils import create_model
from ..transform import WeakLabelAdapter
from ..utils.sk_utils import create_weak_splits

class EndModelExperiment(ExperimentPipeline):
    """Basic end model experiment pipeline"""
    
    def __init__(self, label_model='majority', end_model='linear', cv_folds=5):
        super().__init__()
        self.label_model = label_model
        self.end_model = end_model
        self.cv_folds = cv_folds
        self.pipeline_ = None
    
    def run_experiment(self, X, L, y, param_grid=None, **experiment_params):
        """Run end model experiment"""
        
        # Create pipeline
        self.pipeline_ = self._create_pipeline()
        
        # Split data
        splits = create_weak_splits(X, L, y, **experiment_params)
        
        # Fit label model and get soft labels
        label_model = self._get_label_model()
        label_model.fit(splits['L_train'])
        y_soft_train = label_model.transform(splits['L_train'])
        y_soft_val = label_model.transform(splits['L_val'])
        
        # Train end model
        end_model = self._get_end_model()
        end_model.fit(splits['X_train'], np.argmax(y_soft_train, axis=1))
        
        # Evaluate
        y_pred_val = end_model.predict(splits['X_val'])
        y_pred_test = end_model.predict(splits['X_test'])
        
        # Store results
        self.results_ = {
            'val_accuracy': np.mean(y_pred_val == splits['y_val']),
            'test_accuracy': np.mean(y_pred_test == splits['y_test']),
            'label_model': self.label_model,
            'end_model': self.end_model
        }
        
        # Hyperparameter search if specified
        if param_grid is not None:
            self._run_grid_search(splits, param_grid)
        
        return self.results_
    
    def _create_pipeline(self):
        """Create the experiment pipeline"""
        steps = [
            ('adapter', WeakLabelAdapter()),
            ('label_model', self._get_label_model()),
            ('end_model', self._get_end_model())
        ]
        return Pipeline(steps)
    
    def _get_label_model(self) -> LabelModel:
        """Get label model instance"""
        return create_model('label', self.label_model)
    
    def _get_end_model(self) -> EndModel:
        """Get end model instance"""
        return create_model('end', self.end_model)
    
    def get_results(self):
        """Get experiment results"""
        return self.results_
    
    
__all__ = ['EndModelExperiment']