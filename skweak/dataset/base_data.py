"""
WeakData base class and extensions for different data modalities and tasks
"""

from abc import ABC
from typing import List, Dict, Literal, Optional, Union, Tuple, Any, Callable, ClassVar
from dataclasses import dataclass
from upath import UPath
from collections import Counter
from pydantic import BaseModel

import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_train_test_split, KFold
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from snorkel.labeling import LFAnalysis

from ..constant import TaskType, DataType, DATA_SET, get_arg_list, fetch_first
from ..logging import logger

@dataclass
class InputConfig:
    directory: UPath
    data_type: DataType
    task_type: TaskType
    meta_info: Dict
    
    def __post_init__(self):
        self.directory = UPath(self.directory)
     
def merge_features(a_feature, b_feature):
    """Merge two feature arrays/tensors"""
    if a_feature is not None and b_feature is not None:
        # Handle numpy arrays
        if isinstance(a_feature, np.ndarray) and isinstance(b_feature, np.ndarray):
            return np.vstack([a_feature, b_feature])
        # Handle torch tensors
        elif torch.is_tensor(a_feature) and torch.is_tensor(b_feature):
            return torch.vstack([a_feature, b_feature])
        # Handle mixed types - convert to numpy
        else:
            a_feat = a_feature if isinstance(a_feature, np.ndarray) else a_feature.numpy()
            b_feat = b_feature if isinstance(b_feature, np.ndarray) else b_feature.numpy()
            return np.vstack([a_feat, b_feat])
    elif a_feature is not None:
        return a_feature
    elif b_feature is not None:
        return b_feature
    else:
        return None
    
def array_to_marginals(y, cardinality=None):
    class_counts = Counter(y)
    if cardinality is None:
        sorted_counts = np.array([v for k, v in sorted(class_counts.items())])
    else:
        sorted_counts = np.zeros(len(cardinality))
        for i, c in enumerate(cardinality):
            sorted_counts[i] = class_counts.get(c, 0)
    marginal = sorted_counts / sum(sorted_counts)
    return marginal
        
class WeakDataBase(ABC):
    """
    Abstract base class for weak supervision datasets

    Provides a unified interface for different data modalities (text, image, tabular)
    and tasks (classification, sequence labeling, graph classification)
    """
    available_data: ClassVar[List[str]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "Sample") or not issubclass(cls.Sample, BaseModel):
            raise TypeError(
                f"{cls.__name__} must define a Sample class that inherits from BaseModel"
            )

    @classmethod
    def input_args(cls):
        return get_arg_list(cls.__init__)
    
    @classmethod
    def from_input(cls, data: Dict[str, Any]) -> "WeakDataBase":
        """Create WeakData instance from input data"""
        raise NotImplementedError

    @classmethod
    def from_name(cls, name: str) -> Dict[str, "WeakDataBase"]:
        """Load dataset by name from configuration"""
        if name not in cls.available_data:
            raise ValueError(f"'{name}' not in {cls.available_data}")

        config = DATA_SET[name]
        config = InputConfig(**config)
        output = {}

        for stage in ["train", "valid", "test"]:
            try:
                data_path = config.directory / f"{stage}.json"
                data = data_path.read_text()
                dict_data = {k: cls.Sample(**v) for k,v in json.loads(data).items()}
                output[stage] = cls.from_input(dict_data)
            except FileNotFoundError:
                continue

        return output

    def __init__(
        self,
        ids: Optional[List[str | int]] = None,
        features: Optional[np.ndarray] = None,
        weak_labels: Optional[np.ndarray] = None,
        true_labels: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Initialize WeakData instance

        Args:
            ids: List of unique identifiers for each sample
            features: Feature data (format depends on modality)
            weak_labels: Weak label matrix of shape (n_samples, n_lfs)
            true_labels: Ground truth labels (if available)
            metadata: Additional metadata about each sample (e.g., n_lfs)
        """
        self.ids = ids
        if features is not None and len(features) > 0:
            self.features = np.asarray(features)
        else:
            self.features = None
        self.weak_labels = np.asarray(weak_labels)
        self.true_labels = np.asarray(true_labels)
        self.metadata = metadata or []

        # Validate data consistency
        self._validate_data()

    def _validate_data(self):
        """Validate that all data components are consistent"""
        if self.weak_labels is not None:
            n_samples = len(self.weak_labels)

            if self.ids and len(self.ids) != n_samples:
                raise ValueError(
                    f"IDs length ({len(self.ids)}) != samples ({n_samples})"
                )

            if self.true_labels is not None and len(self.true_labels) != n_samples:
                raise ValueError(
                    f"True labels length ({len(self.true_labels)}) != samples ({n_samples})"
                )
            # Additional validation can be implemented by subclasses

    @property
    def n_samples(self) -> int:
        """Return number of samples"""
        if self.weak_labels is not None:
            return len(self.weak_labels)
        elif self.ids:
            return len(self.ids)
        else:
            return 0

    @property
    def n_lfs(self) -> int:
        """Return number of labeling functions"""
        if self.weak_labels is not None:
            return self.weak_labels.shape[1] if self.weak_labels.ndim > 1 else 1
        return 0

    @property
    def n_classes(self) -> int:
        """Return number of classes"""
        if self.true_labels is not None:
            return len(np.unique(self.true_labels[self.true_labels >= 0]))
        elif self.weak_labels is not None:
            unique_labels = np.unique(self.weak_labels[self.weak_labels >= 0])
            return len(unique_labels)
        return 0
    
    @property
    def n_features(self) -> int:
        if self.features is not None:
            return self.features.shape[-1]
        return 0


    # ==========================================================================
    # Analysis Methods
    # ==========================================================================
    
    @property
    def lf_summary(self):
        """ Generate summary statistics using Snorkel's LFAnalysis """
        L = self.weak_labels
        Y = self.true_labels
        lf_summary = LFAnalysis(L=L).lf_summary(Y=Y)
        return lf_summary
    
    def ci_matrix(self, verbose=False):
        """Conditional Independence Between LFs (from https://github.com/JieyuZ2/wrench/blob/main/wrench/dataset/basedataset.py#L237)"""
        L = self.weak_labels
        Y = self.true_labels
        n, m = L.shape
        lf_cardinality = [sorted(np.unique(L[:, i])) for i in range(m)]

        n_class = len(np.unique(Y))
        c_idx_l = [Y == c for c in range(n_class)]
        c_cnt_l = [np.sum(c_idx) for c_idx in c_idx_l]
        class_marginal = [c_cnt / n for c_cnt in c_cnt_l]

        cond_probs = np.zeros((n_class, m, max(map(len, lf_cardinality))))
        for c, c_idx in enumerate(c_idx_l):
            for i in range(m):
                card_i = lf_cardinality[i]
                cond_probs[c, i][:len(card_i)] = array_to_marginals(L[:, i][c_idx], card_i)

        cmi_matrix = -np.ones((m, m)) * np.inf
        for i in range(m):
            L_i = L[:, i]
            card_i = lf_cardinality[i]
            for j in range(i + 1, m):
                L_j = L[:, j]
                card_j = lf_cardinality[j]

                cmi_ij = 0.0
                for c, (c_idx, n_c) in enumerate(zip(c_idx_l, c_cnt_l)):
                    cmi = 0.0
                    for ci_idx, ci in enumerate(card_i):
                        for cj_idx, cj in enumerate(card_j):
                            p = np.sum(np.logical_and(L_i[c_idx] == ci, L_j[c_idx] == cj)) / n_c
                            if p > 0:
                                cur = p * np.log(p / (cond_probs[c, i, ci_idx] * cond_probs[c, j, cj_idx]))
                                cmi += cur

                    cmi_ij += class_marginal[c] * cmi
                cmi_matrix[i, j] = cmi_matrix[j, i] = cmi_ij
        verbose and logger.info(
            """Conditional Mutual Information Matrix Between LFs
            CMI(λᵢ; λⱼ | Y) = ∑_c P(Y=c) ∑_{vᵢ,vⱼ} [ 
                P(λᵢ=vᵢ, λⱼ=vⱼ | Y=c) 
                × log[ P(λᵢ=vᵢ, λⱼ=vⱼ | Y=c) / (P(λᵢ=vᵢ | Y=c)P(λⱼ=vⱼ | Y=c)) ]
            ]
            PS: if CMI = 0, λᵢ and λⱼ are conditionally independent given Class
        """
        )
        return cmi_matrix


    def summary(self, include_plots: bool = False) -> pd.DataFrame:
        """
        Generate summary statistics using Snorkel's LFAnalysis
        Returns:
            DataFrame with labeling function statistics
        """
        pass
    
    def agreement_matrix(self, verbose=False) -> np.ndarray:
        """ Compute agreement matrix between labeling functions  (n_lfs, n_lfs) """ 
        n_lfs = self.n_lfs
        agreement = np.zeros((n_lfs, n_lfs))
        for i in range(n_lfs):
            for j in range(n_lfs):
                if i == j:
                    agreement[i, j] = 1.0  # Perfect self-agreement
                else:
                    # Find samples where both LFs provide labels
                    mask = (self.weak_labels[:, i] != -1) & (self.weak_labels[:, j] != -1)
                    if np.any(mask):
                        agreement[i, j] = np.mean(
                            self.weak_labels[mask, i] == self.weak_labels[mask, j]
                        )
                    else:
                        agreement[i, j] = -1 # no overlap
        verbose and logger.info(
        """Agreement Matrix Between LFs
            1: fully agree;  -1: no overlap at all
        """)
        return agreement    
    

    def silhouette_score(
        self, metric: str = "euclidean", use_lf: bool = True
    ) -> Union[float, np.ndarray]:
        """
        Calculate silhouette score using either LF votes or ground truth labels
        Returns:
            If use_lf=True: Array of silhouette scores for each LF
            If use_lf=False: Single silhouette score using ground truth
        Silhouette Score:
            if score = 1, well classified all points far from other clusters
            if score < 0, likely misclassified, points closer to a different cluster
        """
        X = self.features
        if X is None or len(X) < 2:
            return np.array([]) if use_lf else 0.0
        if use_lf:
            L = self.weak_labels
            if L is None:
                return np.array([])
            scores = []
            for lf_idx in range(L.shape[1]):
                lf_labels = L[:, lf_idx]
                mask = lf_labels != -1  # Non-abstaining samples
                if np.sum(mask) < 2 or len(np.unique(lf_labels[mask])) == 1:
                    scores.append(0.0)
                else:
                    try:
                        score = sklearn_silhouette_score(
                            X[mask], lf_labels[mask], metric=metric
                        )
                        scores.append(score)
                    except ValueError:
                        scores.append(0.0)
            return np.array(scores)
        else:
            Y = self.true_labels
            if Y is None or len(np.unique(Y)) == 1:
                return 0.0
            return sklearn_silhouette_score(X, Y, metric=metric)

    # ==========================================================================
    # Data Filtering Methods
    # ==========================================================================

    def _filter_by_mask(self, mask: np.ndarray) -> "WeakDataBase":
        """Filter data by boolean mask"""
        indices = np.where(mask)[0]
        return self._filter_by_indices(indices)

    def _filter_by_indices(self, indices: np.ndarray) -> "WeakDataBase":
        """Filter data by indices"""
        new_ids = [self.ids[i] for i in indices] if self.ids else None
        new_wls = self.weak_labels[indices] if self.weak_labels is not None else None
        new_tls = self.true_labels[indices] if self.true_labels is not None else None
        new_features = self.features[indices] if self.features is not None else None
        new_metadata = [self.metadata[i] for i in indices] if self.metadata else None
        
        return type(self)(
            ids=new_ids,
            features=new_features,
            weak_labels=new_wls,
            true_labels=new_tls,
            metadata=new_metadata
        )


    def filter_abstain_all(self) -> "WeakDataBase":
        """ Remove samples where all labeling functions abstain """
        mask = np.any(self.weak_labels != -1, axis=1)
        return self._filter_by_mask(mask)

    def filter_no_majority(self, min_agreement: int = 2) -> "WeakDataBase":
        """ Remove samples where no clear majority exists among LFs """
        keep_mask = []
        for i in range(self.n_samples):
            labels = self.weak_labels[i]
            non_abstain = labels[labels != -1]
            if len(non_abstain) == 0:
                keep_mask.append(False)
                continue
            # Check if any label appears at least min_agreement times
            unique, counts = np.unique(non_abstain, return_counts=True)
            has_majority = np.any(counts >= min_agreement)
            keep_mask.append(has_majority)

        return self._filter_by_mask(np.array(keep_mask))


    def filter_custom(self, filter_fn: Callable) -> "WeakDataBase":
        """
        Apply custom filtering function on weakly labels and features

        Args:
            filter_fn: Function that takes (features, weak_labels)
              and returns boolean mask

        Returns:
            New WeakData instance with filtered samples
        """
        mask = filter_fn( features=self.features, weak_labels=self.weak_labels)
        return self._filter_by_mask(mask)

    # ==========================================================================
    # Data Splitting Methods
    # ==========================================================================

    def split_by_ids(
        self, ids: List[str | int]
    ) -> Tuple["WeakDataBase", "WeakDataBase"]:
        """Split data based on provided IDs"""
        if not self.ids:
            raise ValueError("No IDs available for splitting")
        mask = np.array([id_ in ids for id_ in self.ids])
        selected = self._filter_by_mask(mask)
        remaining = self._filter_by_mask(~mask)
        
        return selected, remaining

    def train_test_split(
        self,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: Optional[int] = None,
    ) -> Tuple["WeakDataBase", "WeakDataBase"]:
        """Split data into train and test sets"""
        indices = np.arange(self.n_samples)
        
        stratify_labels = None
        if stratify and self.true_labels is not None:
            stratify_labels = self.true_labels
        
        train_idx, test_idx = sklearn_train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state
        )
        train_data = self._filter_by_indices(train_idx)
        test_data = self._filter_by_indices(test_idx)
        return train_data, test_data

    def k_fold_split(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> List[Tuple["WeakDataBase", "WeakDataBase"]]:
        """Generate k-fold splits"""
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits = []
        for train_idx, val_idx in kfold.split(range(self.n_samples)):
            train_data = self._filter_by_indices(train_idx)
            val_data = self._filter_by_indices(val_idx)
            splits.append((train_data, val_data))
        return splits

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> "WeakDataBase":
        """Sample subset of data"""
        if n is None and frac is None:
            raise ValueError("Either n or frac must be specified")
        np.random.seed(random_state)
        if frac is not None:
            n = int(frac * self.n_samples)
        if replace:
            indices = np.random.choice(self.n_samples, size=n, replace=True)
        else:
            indices = np.random.choice(self.n_samples, size=min(n, self.n_samples), replace=False)
        return self._filter_by_indices(indices)
    
    # ==========================================================================
    # Utility Methods
    # ==========================================================================


    def clone(self) -> "WeakDataBase":
        """Create a deep copy of the dataset"""
        from copy import deepcopy
        return deepcopy(self)

    def merge(self, other: "WeakDataBase") -> "WeakDataBase":
        """Merge with another WeakData instance"""
        # Basic implementation - can be enhanced
        if type(self) is not type(other):
            raise TypeError("Cannot merge different WeakData types")
        # Combine ids
        combined_ids = (self.ids or []) + (other.ids or [])
        # Merge weaklabels
        combined_weak_labels = None
        if self.weak_labels is not None and other.weak_labels is not None:
            combined_weak_labels = np.vstack([self.weak_labels, other.weak_labels])
        # Merge truelabels
        combined_true_labels = None
        if self.true_labels is not None and other.true_labels is not None:
            combined_true_labels = np.hstack([self.true_labels, other.true_labels])
        # Merge features
        combined_features = merge_features(self.features, other.features)
        # Merge metadata
        combined_metadata =  self.metadata +other.metadata
        
        return type(self)(
            ids=combined_ids,
            features=combined_features,
            weak_labels=combined_weak_labels,
            true_labels=combined_true_labels,
            metadata=combined_metadata
        )
        
    def save(self, path: str, format: str = "pickle") -> None:
        """ Save dataset to disk """
        path = UPath(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path.as_posix())

    @classmethod
    def load(cls, path: str, format: str = "pickle") -> "WeakDataBase":
        """ Load dataset from disk """
        return torch.load(path, map_location='cpu')

    def __len__(self) -> int:
        """Return number of samples"""
        return self.n_samples

    def __getitem__(self, key) -> "WeakDataBase":
        """Support indexing and slicing"""
        if isinstance(key, int):
            return self._filter_by_indices([key])
        elif isinstance(key, slice):
            indices = list(range(self.n_samples))[key]
            return self._filter_by_indices(indices)
        elif isinstance(key, (list, np.ndarray)):
            return self._filter_by_indices(key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def to_record(
        self, orient: Literal['records', 'dict'] = "records"
    ) -> Union[List[Dict], Dict[str, List | np.ndarray]]:
        """
        Convert instance data to record format
        Returns:
            List of records or dictionary of lists based on orient
        """
        if orient == "records":
            # List of dictionaries (one per sample)
            keys = self.input_args()
            return [{
                k: fetch_first(getattr(o, k)) for k in keys
            } for o in iter(self)]
        elif orient == "dict":
            # Dictionary of lists/arrays
            return {
                k: getattr(self, k) 
                for k in self.input_args()
            }
        else:
            raise ValueError("orient must be 'records' or 'dict'") 

    def get_sample_by_id(self, sample_id: str | int) -> Dict:
        """Get sample data by ID"""
        try:
            idx = self.ids.index(sample_id)
            return self.__getitem__(idx)
        except ValueError:
            raise KeyError(f"Sample ID {sample_id} not found")

    def __repr__(self) -> str:
        """String representation of the dataset"""
        parts = [f"n_samples={self.n_samples}"]
        if self.n_features > 0:
            parts.append(f"n_features={self.n_features}")
        parts.extend([f"n_lfs={self.n_lfs}", f"n_classes={self.n_classes}"])
        return f"Sample{self.__class__.__name__}({', '.join(parts)})"
        
