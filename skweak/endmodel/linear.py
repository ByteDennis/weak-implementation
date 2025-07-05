import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array
from ..core import EndModel, register_end_model

@register_end_model('Linear', aliases=['linear'])
class LinearModel(EndModel):
    """Linear model for weak supervision"""

    def __init__(self, C=1.0, random_state=None, **kwargs):
        super().__init__()
        self.C = C
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y, sample_weight=None, **fit_params):
        X, y = check_X_y(X, y, accept_sparse=True)

        self.model_ = LogisticRegression(
            C=self.C, random_state=self.random_state, **self.kwargs
        )

        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.feature_dim_ = X.shape[1]

        return self

    def predict(self, X):
        check_array(X, accept_sparse=True)
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_array(X, accept_sparse=True)
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        check_array(X, accept_sparse=True)
        return self.model_.decision_function(X)
