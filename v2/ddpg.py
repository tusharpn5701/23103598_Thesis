"""
Deep Deterministic Policy Gradient agent (v2).

Fixes relative to the thesis code
---------------------------------
1. Action scaling. The actor ends in tanh (range -1..1). The thesis attacker
   clipped that to 0..1 and the keeper to -1..1, and the environment then read
   the numbers as metres, so every shot landed in the bottom-left 1 m x 1 m of
   the goal and the keeper never left that corner. v2 maps tanh output linearly
   onto the real action box (the goal, plus a small margin for the attacker so
   it can learn not to shoot wide).
2. The critic target uses (1 - done); each shot is terminal, so the critic
   learns the expected outcome of a shot instead of bootstrapping off a state
   that never changes.
3. Exploration noise decays over training and evaluation is noise-free.
4. Replay uses a NumPy ring buffer. The thesis converted a Python list of
   arrays to a tensor on every update, which is very slow in recent PyTorch.
Layer sizes (64-64), learning rates (actor 1e-4, critic 1e-3), tau = 0.005,
gamma = 0.99 and Gaussian exploration are kept from the thesis.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from replay import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, s):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, s, a):
        x = torch.relu(self.fc1(torch.cat([s, a], 1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    def __init__(self, state_dim, action_low, action_high, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, batch_size=64, memory=20000,
                 noise_start=0.4, noise_end=0.05, noise_decay_steps=15000, seed=0, device="cpu"):
        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.low = np.asarray(action_low, dtype=np.float64)
        self.high = np.asarray(action_high, dtype=np.float64)
        self.action_dim = len(self.low)
        self.device = device
        self.actor = Actor(state_dim, self.action_dim).to(device)
        self.actor_target = Actor(state_dim, self.action_dim).to(device)
        self.critic = Critic(state_dim, self.action_dim).to(device)
        self.critic_target = Critic(state_dim, self.action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.memory = ReplayBuffer(memory, state_dim, (len(action_low),), np.float32, seed)
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.noise_start, self.noise_end, self.noise_decay_steps = noise_start, noise_end, noise_decay_steps
        self.steps = 0
        self.soft_update(1.0)

    # ---- scaling between tanh space [-1, 1] and the real action box (metres)
    def to_env(self, a_tanh):
        return self.low + (np.asarray(a_tanh) + 1.0) * 0.5 * (self.high - self.low)

    def to_tanh(self, a_env):
        return (np.asarray(a_env) - self.low) / (self.high - self.low) * 2.0 - 1.0

    @property
    def noise_std(self):
        frac = min(1.0, self.steps / self.noise_decay_steps)
        return self.noise_start + frac * (self.noise_end - self.noise_start)

    def soft_update(self, tau=None):
        tau = self.tau if tau is None else tau
        for net, tgt in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
            for tp, p in zip(tgt.parameters(), net.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def act(self, state, explore=True):
        with torch.no_grad():
            a = self.actor(torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
        a = a.cpu().numpy()[0]
        if explore:
            a = a + self.rng.normal(0, self.noise_std, size=self.action_dim)
        a = np.clip(a, -1.0, 1.0)
        return self.to_env(a), a          # (metres, tanh-space action stored in replay)

    def q_value(self, state, a_env) -> float:
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        a = torch.as_tensor(self.to_tanh(a_env), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self.critic(s, a).item())

    def remember(self, state, a_tanh, reward, next_state, done=True):
        self.memory.add(state, a_tanh, reward, next_state, done)

    def replay(self):
        self.steps += 1
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            q_next = self.critic_target(s2, self.actor_target(s2))
            q_target = r + self.gamma * (1.0 - d) * q_next
        critic_loss = nn.functional.mse_loss(self.critic(s, a), q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()
        return float(critic_loss.item())
