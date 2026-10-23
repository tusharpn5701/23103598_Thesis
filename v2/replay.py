"""Fixed-size replay buffer backed by NumPy arrays (O(1) sampling)."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_shape=(), action_dtype=np.int64, seed=0):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, *action_shape), dtype=action_dtype)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.i = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, float(done)
        self.i = (j + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = self.rng.integers(0, self.size, size=n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

    def __len__(self):
        return self.size
