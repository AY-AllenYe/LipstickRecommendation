import numpy as np
import jittor as jt
from collections import Counter

def compute_class_weights(label_file, num_classes):
    with open(label_file, 'r') as f:
        labels = [int(line.strip().split()[1]) for line in f.readlines()]
    count = Counter(labels)
    total = sum(count.values())
    weights = [total / (count[i] + 1e-5) for i in range(num_classes)]
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize
    return jt.float32(weights)
