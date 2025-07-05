import numpy as np

from ..core import LabelModel, register_label_model
from ..utils.validation import check_weak_labels, get_classes_from_weak_labels

@register_label_model('Majority Voting', aliases=['mv'])
class MajorityVoting(LabelModel):
    """Majority voting label model"""

    def __init__(self, abstain_value=-1):
        super().__init__()
        self.abstain_value = abstain_value

    def fit(self, L, y=None, **fit_params):
        L = check_weak_labels(L)
        self.classes_ = get_classes_from_weak_labels(L, self.abstain_value)
        self.n_classes_ = len(self.classes_)
        self.n_sources_ = L.shape[1]
        return self

    def transform(self, L):
        L = check_weak_labels(L)
        n_samples = L.shape[0]
        probs = np.zeros((n_samples, self.n_classes_))

        for i in range(n_samples):
            votes = L[i][L[i] != self.abstain_value]
            if len(votes) > 0:
                unique, counts = np.unique(votes, return_counts=True)
                for label, count in zip(unique, counts):
                    if label in self.classes_:
                        class_idx = np.where(self.classes_ == label)[0][0]
                        probs[i, class_idx] = count

        # Normalize
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        probs = probs / row_sums

        return probs
