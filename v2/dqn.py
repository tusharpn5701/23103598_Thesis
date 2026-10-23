"""
Deep Q-Network agent (v2).

Fixes relative to the thesis code
---------------------------------
1. A Q-network has to output one value per *discrete* action and the agent
   picks the argmax. The thesis network output two numbers that were used
   directly as the (x, y) aim while the loss regressed only output[0] towards
   the reward, so the action taken never entered the loss. v2 discretises the
   goal into a grid of aim points (Chapter 5.4 of the thesis already describes
   DQN as "discretising actions into a limited set") and learns Q(s, a) for
   every cell.
2. Epsilon now decays (1.0 -> 0.05). The thesis increased epsilon from 0.1 to
   1.0, which meant that late in training, and at evaluation time, almost every
   action was a uniformly random point inside the goal.
3. Evaluation is greedy (epsilon = 0) and the network is put in eval() mode so
   dropout is off.
4. Replay is vectorised (one batched update instead of 32 single-sample
   updates, sampled from a NumPy ring buffer) and uses the TD target r + gamma * (1 - done) * max Q_target.
   Each shot is a one-step episode, so done = 1 and the target is the reward.
The network architecture (64-64-32-32 with dropout 0.2), RAdam, gamma = 0.95
and the target network are kept from the thesis.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from replay import ReplayBuffer


class QNetwork(nn.Module):
    """Same layer sizes as the thesis; the last layer now has one output per action."""

    def __init__(self, state_dim: int, n_actions: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, n_actions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        return self.fc5(x)


def make_grid(u_lo, u_hi, nu, v_lo, v_hi, nv) -> np.ndarray:
    """Discrete action set: an (nu * nv, 2) array of points on the goal plane."""
    us = np.linspace(u_lo, u_hi, nu)
    vs = np.linspace(v_lo, v_hi, nv)
    return np.array([[u, v] for v in vs for u in us], dtype=np.float64)


class DQNAgent:
    def __init__(self, state_dim, action_grid, lr=5e-4, gamma=0.95, batch_size=64,
                 memory=20000, eps_start=1.0, eps_end=0.05, eps_decay_steps=15000,
                 target_update=200, seed=0, device="cpu"):
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.grid = np.asarray(action_grid)
        self.n_actions = len(self.grid)
        self.device = device
        self.model = QNetwork(state_dim, self.n_actions).to(device)
        self.target_model = QNetwork(state_dim, self.n_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory, state_dim, (), np.int64, seed)
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.target_update = target_update
        self.steps = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def q_values(self, state) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            q = self.model(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
        return q.cpu().numpy()[0]

    def act_index(self, state, explore=True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def act(self, state, explore=True):
        idx = self.act_index(state, explore)
        return self.grid[idx].copy(), idx

    def remember(self, state, action_idx, reward, next_state, done=True):
        self.memory.add(state, action_idx, reward, next_state, done)

    def replay(self):
        self.steps += 1
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.int64, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)

        self.model.train()
        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target = r + self.gamma * (1.0 - d) * self.target_model(s2).max(1)[0]
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return float(loss.item())
