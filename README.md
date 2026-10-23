# Attacker and Goalkeeper Agents in Simulated Soccer

Code for the MSc thesis *A Comparative Analysis of Reinforcement Learning Approaches for Attacker and Goalkeeper Agents in Simulated Soccer* (Tushar Purushothaman Namboodiri, University of Galway, 2024, supervised by Dr. Enda Howley).

Two attackers and two goalkeepers are trained, one of each with a Deep Q-Network (DQN) and one with Deep Deterministic Policy Gradient (DDPG), then played against each other from inside the penalty box.

**Interactive site:** open `docs/index.html` in a browser, or enable GitHub Pages on the `docs/` folder. The trained networks run in the page, so you can watch any attacker face any keeper, see each network's estimated scoring chance over the goal, and take penalties yourself.

## Repository layout

```
*.py                 original thesis scripts, unchanged (written as notebook cells)
v2/                  cleaned-up, runnable version used for the site
  soccer_env.py      one shared penalty-box environment
  dqn.py             DQN agent (discrete aim grid)
  ddpg.py            DDPG agent (continuous aim)
  replay.py          NumPy replay buffer
  agents.py          attacker / keeper wrappers around both agents
  train.py           trains all four agents (thesis Ch. 5 and 6)
  evaluate.py        head-to-head matrix and figures (thesis Ch. 7)
  export_web.py      exports weights and curves for the site
  models/            trained weights
  results/           curves.json, results.json
  figures/           learning curves and 100-shot simulation plots
docs/index.html      the interactive site
site/                template + build script for docs/index.html (python site/build_site.py)
```

## Running v2

```
pip install -r v2/requirements.txt
cd v2
python train.py          # all four agents, about 10 minutes on a laptop CPU, including evaluation
python evaluate.py       # results/results.json and figures/
python export_web.py     # web/data.json for the site
```

## What changed in v2

The thesis architectures and hyperparameters are kept (DQN 64-64-32-32 with dropout and RAdam, DDPG 64-64 actor and critic with soft target updates). The changes fix bugs and make the environment a real contest between shooter and keeper.

1. **Every script runs.** Fixed the syntax errors in `DDPG_attacker.py` and `DQN_Simulation.py`, the `_calculate_reward` / `calculate_reward` mismatch, the `SoccerEnvphase3` / `SoccerEnvPhase3` name, and the missing imports in phases 2 and 3.
2. **DQN uses discrete actions.** The Q-network outputs one value per aim point on a 15 x 5 grid and the agent takes the argmax; the loss is on Q(s, a) for the action taken. The original network output an (x, y) pair that was used directly as the action, with the loss applied only to its first output.
3. **Exploration decays** from 1.0 to 0.05. The original epsilon rose from 0.1 to 1.0, so late-training and evaluation actions were uniformly random points inside the goal.
4. **DDPG actions are scaled** from tanh (-1..1) to goal coordinates. Previously they were clipped and read as metres, so every attacker shot and keeper position stayed inside a 1 m square at the left post.
5. **Terminal transitions.** Each shot ends its episode, so the TD target is the reward instead of bootstrapping from an unchanged state.
6. **A keeper that has to move.** The keeper starts centrally and can travel only as far as reaction time (0.18 s) and dive speed (5 m/s) allow during the ball's flight. It sees a noisy read of the shot (0.35 m error) rather than the attacker's exact target.
7. **Shots that can miss.** Ball placement has execution error that grows with distance and angle, so aiming for a corner trades keeper reach against missing the frame.
8. **Shooter position matters.** The attacker state is its spot in the box (6 to 16.5 m out, up to 10 m either side of centre), which sets both shot error and keeper time.
9. **One environment.** Training, evaluation and the head-to-head simulation share one `PenaltyEnv`. The save radius (1.5 m, 1.0 m and 0.5 m in different original files) is 1.0 m everywhere.
10. **Greedy evaluation** with exploration and dropout off, over 5,000 shots per matchup.

## v2 results

Goal % per matchup, 5,000 shots each from random spots in the box (saved % in brackets).

| Attacker | No keeper | Scripted keeper | DQN keeper | DDPG keeper |
|---|---|---|---|---|
| DQN attacker | 90.8 (0.0) | 65.4 (24.3) | 75.6 (14.1) | 66.2 (23.5) |
| DDPG attacker | 84.0 (0.0) | 72.5 (11.8) | 75.1 (9.2) | 72.7 (11.6) |
| Random-aim attacker | 88.3 (0.0) | 28.0 (60.8) | 39.9 (48.5) | 28.2 (60.9) |

Keeper save rate on shots that were on target:

| Keeper | vs random-aim attacker | vs DQN attacker | vs DDPG attacker |
|---|---|---|---|
| Scripted keeper | 68.5% | 27.0% | 14.0% |
| DQN keeper | 54.9% | 15.7% | 10.9% |
| DDPG keeper | 68.4% | 26.2% | 13.8% |

The thesis figures (attackers 100% on an empty goal; keepers 63% and 99.9% vs a random-aim attacker; 55% and 71% vs the same-algorithm attacker) come from the original environment, so they are a different experiment from the table above rather than a baseline it improves on.
