import pytest
import skweak.dataset as Data
from skweak.core import registry
from skweak.labelmodel import MajorityWeightedVoting
from skweak.constant import set_pdb


@pytest.fixture
def train_data():
    data = Data.WeakTextData.from_name("agnews")
    return data["valid"]

@pytest.fixture
def test_data():
    data = Data.WeakTextData.from_name("agnews")
    return data["test"]

@pytest.mark.labelmodel
def test_majority_voting(train_data, test_data):
    for m in (
        MajorityWeightedVoting(abstain_value=-1),
        registry.get_model("MWV")(abstain_value=-1),
        registry.get_model("MajorityWeightedVoting")(abstain_value=-1),
    ):
        m.fit(train_data, balance=None)
        p = m.predict_proba(test_data)
        assert m.balance.tolist() == [0.5] * 4
        assert p.shape == (train_data.n_samples, train_data.n_class) 
        assert (p.sum(axis=1) == 1).all()


