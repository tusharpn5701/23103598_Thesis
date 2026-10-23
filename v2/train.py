"""
Train the four agents (thesis Chapters 5 and 6).

    python train.py                                  # all four agents, default settings
    python train.py --attacker-episodes 1000 --keeper-episodes 2000   # quicker run

Following the thesis, each agent is trained in isolation:
  * attackers shoot against a scripted keeper (dives at its read of the ball)
  * keepers face the thesis' "simplistic attacker" (uniform random aim)
An episode is 10 shots, as in the thesis. Keepers get more episodes than
attackers: their state is 4-D and they need more data. Exploration decays over
the first 60% of training. Models go to models/, curves to results/curves.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from soccer_env import (PenaltyEnv, GOAL, SAVED, MISSED, attacker_reward, keeper_reward,
                        SimplisticAttacker, HeuristicKeeper)
from agents import DQNAttacker, DDPGAttacker, DQNKeeper, DDPGKeeper

SHOTS_PER_EPISODE = 10


def train_attacker(attacker, episodes, seed):
    env = PenaltyEnv(seed)
    keeper = HeuristicKeeper()
    log = {"reward": [], "goals": [], "saved": [], "missed": []}
    for ep in range(episodes):
        tot, counts = 0.0, [0, 0, 0]
        for _ in range(SHOTS_PER_EPISODE):
            lat, dist = env.reset()
            aim = attacker.act(lat, dist, explore=True)
            ball = env.shoot(aim)
            outcome, _ = env.resolve(ball, keeper.act(env.keeper_read(ball), lat, dist))
            r = attacker_reward(outcome)
            attacker.learn(r)
            tot += r
            counts[outcome] += 1
        log["reward"].append(tot)
        log["goals"].append(counts[GOAL])
        log["saved"].append(counts[SAVED])
        log["missed"].append(counts[MISSED])
        if ep % 500 == 0:
            print(f"  [{attacker.name} attacker] ep {ep:5d}  reward {tot:7.1f}  goals {counts[GOAL]}/10")
    return log


def train_keeper(keeper, episodes, seed):
    env = PenaltyEnv(seed)
    shooter = SimplisticAttacker(seed + 1)
    log = {"reward": [], "saved": [], "goals": [], "missed": []}
    for ep in range(episodes):
        tot, counts = 0.0, [0, 0, 0]
        for _ in range(SHOTS_PER_EPISODE):
            lat, dist = env.reset()
            ball = env.shoot(shooter.act(lat, dist))
            target = keeper.act(env.keeper_read(ball), lat, dist, explore=True)
            outcome, gk = env.resolve(ball, target)
            r = keeper_reward(outcome, ball, gk)
            keeper.learn(r)
            tot += r
            counts[outcome] += 1
        log["reward"].append(tot)
        log["goals"].append(counts[GOAL])
        log["saved"].append(counts[SAVED])
        log["missed"].append(counts[MISSED])
        if ep % 500 == 0:
            print(f"  [{keeper.name} keeper]   ep {ep:5d}  reward {tot:7.1f}  saves {counts[SAVED]}/10")
    return log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attacker-episodes", type=int, default=4000)
    p.add_argument("--keeper-episodes", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=".")
    args = p.parse_args()
    torch.set_num_threads(1)          # the networks are tiny; extra threads only add overhead
    na, nk = args.attacker_episodes, args.keeper_episodes
    decay_a, decay_k = int(0.6 * na * SHOTS_PER_EPISODE), int(0.6 * nk * SHOTS_PER_EPISODE)

    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    curves = {}
    jobs = [
        ("attacker_dqn", DQNAttacker(seed=args.seed, eps_decay_steps=decay_a), train_attacker, na),
        ("attacker_ddpg", DDPGAttacker(seed=args.seed, noise_decay_steps=decay_a), train_attacker, na),
        ("keeper_dqn", DQNKeeper(seed=args.seed, lr=1e-3, eps_decay_steps=decay_k), train_keeper, nk),
        ("keeper_ddpg", DDPGKeeper(seed=args.seed, noise_decay_steps=decay_k), train_keeper, nk),
    ]
    for name, agent, fn, episodes in jobs:
        t0 = time.time()
        print(f"Training {name} for {episodes} episodes")
        curves[name] = fn(agent, episodes, args.seed)
        agent.save(os.path.join(args.out, "models", f"{name}.pth"))
        print(f"  done in {time.time() - t0:.0f}s")

    with open(os.path.join(args.out, "results", "curves.json"), "w") as f:
        json.dump({"shots_per_episode": SHOTS_PER_EPISODE, "seed": args.seed, "curves": curves}, f)


if __name__ == "__main__":
    main()
