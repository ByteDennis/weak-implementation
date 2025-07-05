import numpy as np
from sklearn.preprocessing import LabelEncoder


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
