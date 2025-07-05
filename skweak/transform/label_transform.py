import numpy as np
from sklearn.preprocessing import LabelEncoder
from ..core.base import WeakTransformer
from ..utils.validation import check_weak_labels


class WeakLabelAdapter(WeakTransformer):
    """Adapter for standardizing weak label formats"""

    def __init__(self, abstain_value=-1, output_format="matrix"):
        self.abstain_value = abstain_value
        self.output_format = output_format
        self.label_encoder_ = None
        self.n_sources_ = None
        self.classes_ = None

    def fit(self, L, y=None):
        L = check_weak_labels(L)

        self.n_sources_ = L.shape[1] if L.ndim > 1 else 1

        # Get unique labels excluding abstain
        mask = L != self.abstain_value
        unique_labels = np.unique(L[mask])
        self.classes_ = np.sort(unique_labels)

        # Fit label encoder
        self.label_encoder_ = LabelEncoder()
        if len(unique_labels) > 0:
            self.label_encoder_.fit(unique_labels)

        return self

    def transform(self, L):
        L = check_weak_labels(L)

        if self.output_format == "matrix":
            return self._to_matrix_format(L)
        elif self.output_format == "encoded":
            return self._encode_labels(L)
        else:
            return L

    def _to_matrix_format(self, L):
        """Convert to standard matrix format"""
        if L.ndim == 1:
            L = L.reshape(-1, 1)
        return L

    def _encode_labels(self, L):
        """Encode labels to standard format"""
        L_encoded = L.copy()
        mask = L != self.abstain_value

        if np.any(mask):
            L_encoded[mask] = self.label_encoder_.transform(L[mask])

        return L_encoded

