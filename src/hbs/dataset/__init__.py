"""
Dataset package: synthetic contradiction data and torch loaders.
"""

from hbs.dataset.loader import IGNORE_INDEX, SyntheticContradictionDataset, collate_fn
from hbs.dataset.synthetic import ContradictionSample, build_samples

__all__ = [
    "IGNORE_INDEX",
    "SyntheticContradictionDataset",
    "collate_fn",
    "ContradictionSample",
    "build_samples",
]
