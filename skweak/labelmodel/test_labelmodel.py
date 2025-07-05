import pytest
import skweak.dataset as Data
import numpy as np
from skweak.labelmodel import MajorityVoting
from skweak.utils import create_model, get_label_model

lm = pytest.mark.labelmodel

@pytest.fixture
def train_data():
    data = Data.WeakTextData.from_name("agnews")
    return data["valid"]

@pytest.fixture
def test_data():
    data = Data.WeakTextData.from_name("agnews")
    return data["test"]

@lm
def test_mv(train_data, test_data):
    for m in (
        MajorityVoting(abstain_value=-1),
        create_model("label", "Majority Voting"),
        get_label_model('mv')
    ):
        X_train = train_data.weak_labels.astype(np.float32)
        y_train = train_data.true_labels.astype(np.float32)
        m.fit(X_train, y_train)
        X_test  = test_data.weak_labels.astype(np.float32)
        y_test  = test_data.true_labels.astype(np.float32)
        train_score = m.score(X_train, y_train) 
        test_score  = m.score(X_test, y_test) 
        print(
            f"\n\ntrain score: {train_score:<5.4f}\n"
            f"test score: {test_score:<5.4f}"
        )

