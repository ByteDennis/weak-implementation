import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List
from weakly.experiments.adversarial import AdversarialExperiment


""" 
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Pytest Fixture                                                           │
  └──────────────────────────────────────────────────────────────────────────┘
"""
@pytest.fixture
def base_config():
    # Basic configuration for WMRC experiments
    return {
        "dataset_name": "test_dataset",
        "constraint_form": "accuracy",
        "n_runs": 1,
        "use_test": True,
        "get_confidences": False,
        "log_level": "INFO",
    }


# class TestExperimentInitialization:
    
#     @patch('weakly.utils.logging.ExperimentLogger')
#     @patch('wmrc.experiments.wmrc_experiment.WMRCConfidenceComputer')
#     @patch('wmrc.experiments.wmrc_experiment.MatlabDatasetLoader')