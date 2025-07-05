import torch
import numpy as np
from sklearn.utils import check_random_state


def set_torch_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


def numpy_to_torch(array, dtype=torch.float32, device=None):
    """Convert numpy array to torch tensor"""
    tensor = torch.tensor(array, dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
        
    return tensor


def torch_to_numpy(tensor):
    """Convert torch tensor to numpy array"""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


class EarlyStopping:
    """Early stopping utility for PyTorch training"""

    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model=None):
        """Check if training should stop"""
        if self.best_score is None:
            self.best_score = score
            if model and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if model and self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

        return False
