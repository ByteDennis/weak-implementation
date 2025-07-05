import numpy as np
from ..core.base import WeakTransformer
from sklearn.preprocessing import StandardScaler


class FeatureAdapter(WeakTransformer):
    """Adapter for feature preprocessing"""

    def __init__(self, scaler="standard", feature_selection=None):
        self.scaler = scaler
        self.feature_selection = feature_selection
        self.scaler_ = None
        self.feature_selector_ = None

    def fit(self, X, y=None):
        # Initialize scaler
        if self.scaler == "standard":
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)

        # Initialize feature selector
        if self.feature_selection is not None:
            self.feature_selector_ = self.feature_selection
            self.feature_selector_.fit(X, y)

        return self

    def transform(self, X):
        # Apply scaling
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        # Apply feature selection
        if self.feature_selector_ is not None:
            X = self.feature_selector_.transform(X)

        return X


class MultiModalAdapter(WeakTransformer):
    """Adapter for multi-modal data (text, images, etc.)"""

    def __init__(
        self, text_processor=None, image_processor=None, combine_method="concat"
    ):
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.combine_method = combine_method

    def fit(self, X, y=None):
        # X should be a dict with 'text' and 'image' keys
        if self.text_processor and "text" in X:
            self.text_processor.fit(X["text"], y)

        if self.image_processor and "image" in X:
            self.image_processor.fit(X["image"], y)

        return self

    def transform(self, X):
        features = []

        if self.text_processor and "text" in X:
            text_features = self.text_processor.transform(X["text"])
            features.append(text_features)

        if self.image_processor and "image" in X:
            image_features = self.image_processor.transform(X["image"])
            features.append(image_features)

        if self.combine_method == "concat":
            return np.concatenate(features, axis=1)
        else:
            return features
