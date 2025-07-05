import numpy as np
from sklearn.utils.validation import check_array, check_X_y
from dataclasses import dataclass

def dataclass2tree(obj: dataclass, indent=2):
    lines = []
    prefix = " " * indent
    lines.append(f"{prefix}{obj.__class__.__name__}")
    for field in obj.__dataclass_fields__:
        val = getattr(obj, field)
        if hasattr(val, "__dataclass_fields__"):
            lines.append(f"{prefix}  {field}:")
            lines.append(dataclass2tree(val, indent + 2))
        else:
            lines.append(f"{prefix}  {field}: {val}")
    return "\n".join(lines)

def check_weak_labels(L, accept_sparse=False, dtype=None):
    """Validate weak label matrix"""
    L = check_array(L, accept_sparse=accept_sparse, dtype=dtype)
    return L


def check_X_L(X, L, accept_sparse=False, dtype=None):
    """Validate features and weak labels"""
    X = check_array(X, accept_sparse=accept_sparse, dtype=dtype)
    L = check_weak_labels(L, accept_sparse=accept_sparse, dtype=dtype)

    if X.shape[0] != L.shape[0]:
        raise ValueError("X and L must have the same number of samples")

    return X, L


def validate_label_model_input(L, abstain_value=-1):
    """Validate input for label models"""
    L = check_weak_labels(L)

    # Check for valid weak labels
    unique_vals = np.unique(L)
    if len(unique_vals) < 2:
        raise ValueError("Weak labels must contain at least 2 distinct values")

    return L


def get_classes_from_weak_labels(L, abstain_value=-1):
    """Extract class labels from weak label matrix"""
    unique_vals = np.unique(L)
    classes = unique_vals[unique_vals != abstain_value]
    return np.sort(classes)
