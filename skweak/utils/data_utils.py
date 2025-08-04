import numpy as np
from sklearn.datasets import make_classification
from ..constant import DATA_HOME
from .io_utils import cache


@cache(directory=DATA_HOME / '.cache')
def download_data(force_restart = False):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="jieyuz2/WRENCH", repo_type="dataset", local_dir=DATA_HOME)

download_data()

class WeakLabelGenerator:
    """Generate synthetic weak labels"""

    def __init__(
        self,
        n_sources=5,
        accuracy_range=(0.6, 0.9),
        coverage_range=(0.3, 0.8),
        correlation=0.0,
        random_state=None,
    ):
        self.n_sources = n_sources
        self.accuracy_range = accuracy_range
        self.coverage_range = coverage_range
        self.correlation = correlation
        self.random_state = random_state

        self.source_accuracies_ = None
        self.source_coverages_ = None

    def fit(self, X=None, y=None, **fit_params):
        """Fit generator parameters"""
        np.random.seed(self.random_state)

        # Generate source accuracies and coverages
        self.source_accuracies_ = np.random.uniform(
            self.accuracy_range[0], self.accuracy_range[1], self.n_sources
        )
        self.source_coverages_ = np.random.uniform(
            self.coverage_range[0], self.coverage_range[1], self.n_sources
        )

        return self

    def predict(self, y_true):
        """Generate weak labels from true labels"""
        return self.generate(y_true)

    def generate(self, y_true, return_params=False):
        """Generate weak labels from true labels"""
        n_samples = len(y_true)
        n_classes = len(np.unique(y_true))

        if self.source_accuracies_ is None:
            self.fit()

        # Initialize weak label matrix
        L = np.full((n_samples, self.n_sources), -1, dtype=int)

        for i, (accuracy, coverage) in enumerate(
            zip(self.source_accuracies_, self.source_coverages_)
        ):
            # Determine which samples this source labels
            n_labeled = int(coverage * n_samples)
            labeled_indices = np.random.choice(n_samples, n_labeled, replace=False)

            # Generate labels with specified accuracy
            for idx in labeled_indices:
                if np.random.random() < accuracy:
                    # Correct label
                    L[idx, i] = y_true[idx]
                else:
                    # Random incorrect label
                    incorrect_labels = [c for c in range(n_classes) if c != y_true[idx]]
                    L[idx, i] = np.random.choice(incorrect_labels)

        if return_params:
            return L, {
                "accuracies": self.source_accuracies_,
                "coverages": self.source_coverages_,
            }

        return L



def make_weak_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    n_sources=5,
    accuracy_range=(0.6, 0.9),
    coverage_range=(0.3, 0.8),
    random_state=None,
    **kwargs,
):
    """Generate synthetic weak supervision dataset"""

    # Generate base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=random_state,
        **kwargs,
    )

    # Generate weak labels
    generator = WeakLabelGenerator(
        n_sources=n_sources,
        accuracy_range=accuracy_range,
        coverage_range=coverage_range,
        random_state=random_state,
    )

    L = generator.fit().generate(y)

    return X, L, y

__all__ = ["make_weak_classification"]
