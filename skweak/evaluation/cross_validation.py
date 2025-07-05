import numpy as np
import sklearn.model_selection as ms
from sklearn.base import clone

def cv_score(estimator, X, L, y, cv=5, scoring='accuracy', 
                        use_label_model=True, **kwargs):
    """Cross-validation for weak supervision models"""
    
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        L_train, _      = L[train_idx], L[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone estimator
        estimator_clone = clone(estimator)
        
        if use_label_model:
            # Weak supervision workflow
            estimator_clone.fit(X_train, L_train)
        else:
            # Standard supervised workflow
            estimator_clone.fit(X_train, y_train)
        
        # Score
        if scoring == 'accuracy':
            score = estimator_clone.score(X_test, y_test)
        else:
            y_pred = estimator_clone.predict(X_test)
            if scoring == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y_test, y_pred, average='weighted')
            # Add other metrics as needed
        
        scores.append(score)
    
    return np.array(scores)

class StratifiedKFold:
    """Stratified K-Fold for weak supervision that maintains label distribution"""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.base_cv = ms.StratifiedKFold(n_splits, shuffle, random_state)
    
    def split(self, X, L, y):
        """Generate indices to split data into training and test set"""
        # Use true labels for stratification when available
        # Fall back to majority vote from weak labels if no true labels
        if y is not None:
            stratify_labels = y
        else:
            # Get majority vote from weak labels
            from ..labelmodel import MajorityVoting
            mv = MajorityVoting()
            mv.fit(L)
            stratify_labels = mv.predict(L)
        
        return self.base_cv.split(X, stratify_labels)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits