import numpy as np
from typing import Optional
from collections import Counter


def encode_weak_labels(L, abstain_value=-1):
    """Encode weak labels to standard format"""
    L_encoded = L.copy()

    # Get unique non-abstain values
    mask = L != abstain_value
    unique_labels = np.unique(L[mask])

    # Create mapping
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    label_mapping[abstain_value] = abstain_value

    # Apply mapping
    for old_label, new_label in label_mapping.items():
        L_encoded[L == old_label] = new_label

    return L_encoded, label_mapping


def normalize_probabilities(probs, axis=1):
    """Normalize probability arrays"""
    probs = np.array(probs)
    probs = np.clip(probs, 1e-8, 1 - 1e-8)  # Avoid numerical issues
    return probs / np.sum(probs, axis=axis, keepdims=True)


def soft_to_hard_labels(probs, threshold=0.5):
    """Convert soft labels to hard labels"""
    if probs.ndim == 1:
        return (probs > threshold).astype(int)
    else:
        return np.argmax(probs, axis=1)

def calculate_prior(L: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate class prior from validation labels or weak labels.

    Parameters
    ----------
    L : np.ndarray
        Weak labels matrix of shape (n_samples, n_labeling_functions)
    y_valid : np.ndarray, optional
        True labels for validation
    """
    if y is None:
        y = np.arange(L.max() + 1)
    class_cnts = Counter(y)
    n_class = len(class_cnts)
    sorted_counts = np.zeros(n_class)
    for c, cnt in n_class.items():
        if c < n_class:  # Ensure class index is within bounds
            sorted_counts[c] = cnt
    prior = (sorted_counts + 1) / (sorted_counts.sum() + n_class)
    return prior
