import pytest
import skweak.dataset as Data
from wrench.dataset import BaseDataset
from wrench.dataset.utils import check_weak_labels


@pytest.fixture
def agnews():
    data = Data.WeakTextData.from_name("agnews")
    return data['test']


def test_weakdata_consensus():
    data = Data.WeakData.from_name("census")
    assert isinstance(data["test"][0], Data.WeakData)
    assert isinstance(data["test"][0], BaseDataset)
    assert hasattr(data, 'n_class')
    L = check_weak_labels(data) # must have weak labels?
    
def test_weaktextrl_cdr():
    data = Data.WeakTextRelationData.from_name("cdr")
    assert isinstance(data["test"][0], Data.WeakTextRelationData)
    assert isinstance(data["test"][0], BaseDataset)
    
def test_to_record(agnews):
    x = agnews.to_record(orient='records')
    y = agnews.to_record(orient='dict')
    assert len(x) == 12_000 and x[0]['ids'] == '108000'
    assert len(y) == 6 and y['features'] is None and y['weak_labels'].shape == (12_000, 9)