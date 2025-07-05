import torch
import torch.nn as nn
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.utils.validation import check_X_y, check_array
from ..core import EndModel, register_end_model
from ..utils.torch_utils import set_torch_seed


class MLPModule(nn.Module):
    """Multi-layer perceptron module"""

    def __init__(self, input_size, hidden_sizes=[128, 64], n_classes=2, dropout=0.2):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

@register_end_model('MLP', aliases=['mlp'])
class MLPModel(EndModel):
    """MLP model using skorch for sklearn compatibility"""

    def __init__(
        self,
        hidden_sizes=[128, 64],
        dropout=0.2,
        lr=0.001,
        max_epochs=100,
        batch_size=128,
        random_state=None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y, sample_weight=None, **fit_params):
        X, y = check_X_y(X, y, dtype=np.float32)
        set_torch_seed(self.random_state)

        self.classes_ = np.unique(y)
        self.feature_dim_ = X.shape[1]
        n_classes = len(self.classes_)

        self.model_ = NeuralNetClassifier(
            module=MLPModule,
            module__input_size=self.feature_dim_,
            module__hidden_sizes=self.hidden_sizes,
            module__n_classes=n_classes,
            module__dropout=self.dropout,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            train_split=None,
            verbose=0,
            **self.kwargs,
        )

        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = check_array(X, dtype=np.float32)
        return self.model_.predict(X)

    def predict_proba(self, X):
        X = check_array(X, dtype=np.float32)
        return self.model_.predict_proba(X)
