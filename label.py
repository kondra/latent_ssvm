import numpy as np


class Label(object):
    def __init__(self, full, weak, weights, full_labeled, relaxed=False):
        self.full = full
        self.weak = weak
        self.weights = weights
        self.full_labeled = full_labeled
        self.relaxed = relaxed
        if not self.relaxed:
            if self.full is None and self.weak is None:
                raise ValueError("You should specify full or weak labels")
            if self.weak is None:
                self.weak = np.unique(self.full).astype(np.int32)
            if self.full is None:
                self.full = np.random.choice(self.weak, self.weights.size).astype(np.int32)

    def __eq__(self, other):
        if self.relaxed or other.relaxed:
            return np.all(self.full[0] == other.full[0]) \
                and np.all(self.full[1] == other.full[1])
        return np.all(self.full == other.full)
