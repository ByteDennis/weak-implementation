from typing import List, Optional, Dict, Union, Any, Tuple
from pydantic import BaseModel
import numpy as np
import torch
from .base_data import WeakDataBase


class WeakTextData(WeakDataBase):
    """Base class for weak text data"""
    
    available_data = ["agnews", "youtube", "trec"]
    
    class Sample(BaseModel):
        label: int
        data: Dict[ str, str]   # {text: str}
        weak_labels: List[int]
        
    @classmethod
    def from_input(cls, data):
        """Create WeakTextData from input dictionary"""
        inputs = {k: [] for k in cls.input_args()}
        for _id, _sample in data.items():
            if 'feature' in _sample.data:
                inputs["features"].append(_sample.data['feature'])
            inputs["ids"].append(_id)
            inputs["texts"].append(_sample.data["text"])
            inputs["weak_labels"].append(_sample.weak_labels)
            inputs["true_labels"].append(_sample.label)
            inputs["metadata"].append({})

        return cls(**inputs)
        
    def __init__(
        self,
        ids: Optional[List[str | int]] = None,
        texts: Optional[List[str]] = None,
        features: Optional[
            np.ndarray
        ] = None,  # Pre-computed features (e.g., embeddings)
        weak_labels: Optional[np.ndarray] = None,
        true_labels: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Initialize WeakTextRelationData

        Args:
            ids: List of sample identifiers
            texts: List of text strings with entity markers
            features: Pre-computed text features/embeddings
            weak_labels: Weak supervision labels
            true_labels: Ground truth labels
            metadata: Additional metadata for each sample
        """
        super().__init__(ids, features, weak_labels, true_labels, metadata)
        self.metadata = metadata or []
        self.texts = texts or ''

    def tokenize(
        self, tokenizer: Any = None, max_length: int = 512, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts using provided tokenizer

        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            **kwargs: Additional tokenizer arguments

        Returns:
            Dictionary with tokenized inputs
        """
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        if not self.texts:
            raise ValueError("No texts available for tokenization")

        # Tokenize all texts
        tokenized = tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            **kwargs,
        )
        return tokenized

    def extract_features(
        self, model: Any, tokenizer: Any = None, **kwargs
    ) -> np.ndarray:
        """
        Extract features using a pre-trained model

        Args:
            model: Pre-trained model (e.g., BERT)
            tokenizer: Tokenizer for the model
            **kwargs: Additional arguments

        Returns:
            Extracted features as numpy array
        """
        if not self.texts:
            raise ValueError("No texts available for feature extraction")

        # Tokenize texts
        inputs = self.tokenize(tokenizer, **kwargs)

        # Extract features using the model
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

            # Use [CLS] token representation or pooled output
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                # Use [CLS] token (first token)
                features = outputs.last_hidden_state[:, 0, :]
            else:
                raise ValueError("Model output format not recognized")

        # Store features
        self.features = features.cpu().numpy()
        return self.features
    
    def extract_feature_(self, extract_fn, return_extractor, **kwargs):
        pass
    

class WeakTextRelationData(WeakTextData):
    """Weak Relation Data implementation for text relation classifications between entity1 and entity2"""
    
    available_data = ['cdr']

    class Sample(BaseModel):
        label: int
        data: Dict[ str, str | List[int]
        ]  # {text: str, entity1: str, entity2: str, span1: List[int], span2: List[int]}
        weak_labels: List[int]

    @classmethod
    def from_input(
        cls, data: Dict[str, "WeakTextRelationData.Sample"]
    ) -> "WeakTextRelationData":
        """ Create WeakTextRelationData from input dictionary """
        inputs = {k: [] for k in cls.input_args()}
        for _id, _sample in data.items():
            if 'feature' in _sample.data:
                inputs["features"].append(_sample.data.pop('feature'))
            bert_text = cls._create_bert_text(**_sample.data)
            inputs['texts'].append(bert_text)
            inputs["metadata"].append(_sample.data)
            inputs["ids"].append(_id)
            inputs["weak_labels"].append(_sample.weak_labels)
            inputs["true_labels"].append(_sample.label)
        return cls(**inputs)

    @staticmethod
    def _create_bert_text(
        text: str, entity1: str, entity2: str, span1: List[int], span2: List[int]
    ) -> str:
        """
        Create BERT-style text with entity markers <e1> entity1 </e1> and <e2> entity2 </e2>

        Args:
            text: Original text
            entity1: First entity
            entity2: Second entity
            span1: [start, end] positions for entity1
            span2: [start, end] positions for entity2

        Returns:
            Text with entity markers
        """
        # Sort spans by position to handle insertion order
        spans = [(span1[0], span1[1], 1, entity1), (span2[0], span2[1], 2, entity2)]
        spans.sort(key=lambda x: x[0])

        # Insert markers from right to left to maintain positions
        result = text
        for start, end, entity_num, entity_text in reversed(spans):
            if entity_num == 1:
                result = result[:start] + f"<e1> {entity_text} </e1>" + result[end:]
            else:
                result = result[:start] + f"<e2> {entity_text} </e2>" + result[end:]

        return result

    def get_relation_pairs(self) -> List[Tuple[str, str]]:
        """Get all entity pairs"""
        pairs = []
        for meta in self.metadata:
            pairs.append((meta["entity1"], meta["entity2"]))
        return pairs
