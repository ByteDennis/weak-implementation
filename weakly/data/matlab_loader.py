import os
import numpy as np
import scipy.io as sio
from typing import Dict, Any, Tuple
from .base_loader import DatasetLoader

class MatlabDatasetLoader(DatasetLoader):
    # Loads datasets from Matlab .mat files with WMRC-specific format
    
    def load_dataset(self, dataset_name: str, dataset_prefix: str = "./datasets/") -> Dict:
        # Loads train/val/test data from .mat file format
        dataset_path = os.path.join(dataset_prefix, dataset_name + '.mat')
        return sio.loadmat(dataset_path)
    
    def validate_dataset(self, data: Dict) -> bool:
        # Validates that loaded data has required WMRC format fields
        required_fields = ['train_pred', 'train_labels']
        return all(field in data for field in required_fields)
    
    def split_dataset(self, data: Dict, split_config: Dict) -> Tuple:
        # Extracts weak supervision signals and true labels from data
        train_data = [data['train_pred'], data['train_labels']]
        
        labeled_set = split_config.get('labeled_set', 'valid')
        if labeled_set == 'valid' and 'val_pred' in data:
            labeled_data = [data['val_pred'], data['validation_labels']]
        else:
            labeled_data = train_data
            
        test_data = None
        if split_config.get('use_test', True) and 'test_pred' in data:
            test_data = [data['test_pred'], data['test_labels']]
            
        return train_data, labeled_data, test_data
    
    def _add_majority_vote_constraint(self, weak_labels: Any) -> Any:
        # Adds majority vote as additional weak labeling function
        majority_votes = np.sum(weak_labels, axis=1, keepdims=True)
        return np.concatenate([weak_labels, majority_votes], axis=1)
