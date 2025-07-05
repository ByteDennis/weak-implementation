from typing import List, Dict
from pydantic import BaseModel
from .base_data import WeakDataBase


class WeakData(WeakDataBase):
    """WeakData implementation for tabular data classification"""

    available_data = ["mushroom", "census"]
    
    class Sample(BaseModel):
        label: int
        data: Dict[ str, List[int | float] ] # {feature: List[int]}
        weak_labels: List[int]
    
    @classmethod
    def from_input(cls, data: Dict[str, "WeakData.Sample"]) -> "WeakData":
        """Create WeakData from input dictionary"""
        inputs = {k: [] for k in cls.input_args()}
        for _id, _sample in data.items():
            inputs['ids'].append(_id)
            inputs['features'].append(_sample.data['feature'])
            inputs['weak_labels'].append(_sample.weak_labels)
            inputs['true_labels'].append(_sample.label)
            inputs['metadata'].append({})
        return cls( **inputs )