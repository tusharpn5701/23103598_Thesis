"""
Thin wrappers so every attacker and keeper exposes the same interface:

    attacker.act(lat, dist, explore)        -> aim point (u, v) in metres
    keeper.act(read, lat, dist, explore)    -> target position (u, v) in metres

plus builders that create the four learning agents with the action spaces used
throughout the project.
"""
from __future__ import annotations

import numpy as np
import torch

from soccer_env import GOAL_W, GOAL_H, attacker_obs, keeper_obs
from dqn import DQNAgent, make_grid
from ddpg import DDPGAgent

# Discrete action sets for DQN: 15 x 5 = 75 points on the goal plane
ATTACKER_GRID = make_grid(0.25, GOAL_W - 0.25, 15, 0.2, GOAL_H - 0.2, 5)
KEEPER_GRID = make_grid(0.3, GOAL_W - 0.3, 15, 0.25, GOAL_H - 0.25, 5)

# Continuous action boxes for DDPG (attacker may aim slightly wide so it can learn not to)
ATTACKER_LOW, ATTACKER_HIGH = [-0.5, 0.0], [GOAL_W + 0.5, GOAL_H + 0.3]
KEEPER_LOW, KEEPER_HIGH = [0.0, 0.0], [GOAL_W, GOAL_H]


class DQNAttacker:
    name = "DQN"

    def __init__(self, seed=0, **kw):
        self.agent = DQNAgent(2, ATTACKER_GRID, seed=seed, **kw)
        self._last = None

    def act(self, lat, dist, explore=False):
        s = attacker_obs(lat, dist)
        aim, idx = self.agent.act(s, explore)
        self._last = (s, idx)
        return aim

    def learn(self, reward):
        s, idx = self._last
        self.agent.remember(s, idx, reward, s, done=True)
        return self.agent.replay()

    def save(self, path):
        torch.save(self.agent.model.state_dict(), path)

    def load(self, path):
        self.agent.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.agent.model.eval()


class DDPGAttacker:
    name = "DDPG"

    def __init__(self, seed=0, **kw):
        self.agent = DDPGAgent(2, ATTACKER_LOW, ATTACKER_HIGH, seed=seed, **kw)
        self._last = None

    def act(self, lat, dist, explore=False):
        s = attacker_obs(lat, dist)
        aim, a_t = self.agent.act(s, explore)
        self._last = (s, a_t)
        return aim

    def learn(self, reward):
        s, a_t = self._last
        self.agent.remember(s, a_t, reward, s, done=True)
        return self.agent.replay()

    def save(self, path):
        torch.save({"actor": self.agent.actor.state_dict(), "critic": self.agent.critic.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location="cpu")
        self.agent.actor.load_state_dict(d["actor"])
        self.agent.critic.load_state_dict(d["critic"])


class DQNKeeper:
    name = "DQN"

    def __init__(self, seed=0, **kw):
        self.agent = DQNAgent(4, KEEPER_GRID, seed=seed, **kw)
        self._last = None

    def act(self, read, lat, dist, explore=False):
        s = keeper_obs(read[0], read[1], lat, dist)
        target, idx = self.agent.act(s, explore)
        self._last = (s, idx)
        return target

    def learn(self, reward):
        s, idx = self._last
        self.agent.remember(s, idx, reward, s, done=True)
        return self.agent.replay()

    save = DQNAttacker.save
    load = DQNAttacker.load


class DDPGKeeper:
    name = "DDPG"

    def __init__(self, seed=0, **kw):
        self.agent = DDPGAgent(4, KEEPER_LOW, KEEPER_HIGH, seed=seed, **kw)
        self._last = None

    def act(self, read, lat, dist, explore=False):
        s = keeper_obs(read[0], read[1], lat, dist)
        target, a_t = self.agent.act(s, explore)
        self._last = (s, a_t)
        return target

    def learn(self, reward):
        s, a_t = self._last
        self.agent.remember(s, a_t, reward, s, done=True)
        return self.agent.replay()

    save = DDPGAttacker.save
    load = DDPGAttacker.load


def load_trained(models_dir="models"):
    """Load the four trained agents from disk."""
    import os
    a_dqn, a_ddpg, k_dqn, k_ddpg = DQNAttacker(), DDPGAttacker(), DQNKeeper(), DDPGKeeper()
    a_dqn.load(os.path.join(models_dir, "attacker_dqn.pth"))
    a_ddpg.load(os.path.join(models_dir, "attacker_ddpg.pth"))
    k_dqn.load(os.path.join(models_dir, "keeper_dqn.pth"))
    k_ddpg.load(os.path.join(models_dir, "keeper_ddpg.pth"))
    return {"DQN": a_dqn, "DDPG": a_ddpg}, {"DQN": k_dqn, "DDPG": k_ddpg}
