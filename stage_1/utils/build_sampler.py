import csv
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import math


def build_stage1_sampler(metadata_path, alpha=0.5):
    datasets = []
    
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append(row["dataset"])

    counts = Counter(datasets)

    # temperature scaling
    weights = {d: (1 / (counts[d] ** alpha)) for d in counts}

    sample_weights = [weights[d] for d in datasets]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler
