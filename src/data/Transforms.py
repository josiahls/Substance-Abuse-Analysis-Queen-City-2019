import sys
sys.path.append('../')

from src.data.Analysis import JosiahAnalysis
import torch
import numpy as np


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: dict):
        return {c: torch.from_numpy(np.array(sample[c], dtype=np.float64)).float() for c in sample}


class ToXY(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: dict):
        return {'X': [sample[c] for c in sample if c not in JosiahAnalysis.DECISION_VARIABLES if c in sample
                      if c not in JosiahAnalysis.INDEX_VARIABLES],
                'Y': [sample[c] for c in JosiahAnalysis.DECISION_VARIABLES if c in sample],
                'I': [sample[c] for c in JosiahAnalysis.INDEX_VARIABLES if c in sample]}
