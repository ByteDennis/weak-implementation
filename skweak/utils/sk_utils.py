import numpy as np
from sklearn.model_selection import train_test_split


def create_weak_splits(X, L, y, test_size=0.2, val_size=0.1, random_state=None):
    """Create train/val/test splits for weak supervision"""

    # First split: train+val vs test
    X_temp, X_test, L_temp, L_test, y_temp, y_test = train_test_split(
        X, L, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, L_train, L_val, y_train, y_val = train_test_split(
        X_temp,
        L_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp,
    )

    return {
        "X_train": X_train,
        "L_train": L_train,
        "y_train": y_train,
        "X_val": X_val,
        "L_val": L_val,
        "y_val": y_val,
        "X_test": X_test,
        "L_test": L_test,
        "y_test": y_test,
    }


def combine_weak_and_labeled_data(X_weak, L_weak, X_labeled, y_labeled):
    """Combine weak and labeled datasets"""
    # Implementation for combining datasets
    pass
