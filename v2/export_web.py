"""
Export trained networks, curves and results to one JSON file for the website.

    python export_web.py            # writes web/data.json

The browser re-implements the forward pass of each network (plain matrix
multiplies + ReLU / tanh) and the environment in soccer_env.py, so the
simulation on the site runs the same trained policies as this repository.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from soccer_env import PARAMS, attacker_obs, keeper_obs
from agents import (load_trained, ATTACKER_GRID, KEEPER_GRID, ATTACKER_LOW, ATTACKER_HIGH,
                    KEEPER_LOW, KEEPER_HIGH)


def r5(x):
    return float(f"{x:.5g}")


def linear_layers(module):
    """[[W (out x in), b], ...] for every nn.Linear in definition order."""
    out = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            out.append([[[r5(v) for v in row] for row in m.weight.detach().numpy()],
                        [r5(v) for v in m.bias.detach().numpy()]])
    return out


def smooth(y, window=50, every=10):
    y = np.asarray(y, dtype=float)
    window = max(1, min(window, len(y)))
    ma = np.convolve(y, np.ones(window) / window, mode="valid")
    idx = np.arange(0, len(ma), every)
    return {"x": (idx + window).tolist(), "y": [round(float(v), 3) for v in ma[idx]]}


def main(models_dir="models", results_dir="results", out="web/data.json"):
    attackers, keepers = load_trained(models_dir)
    with open(os.path.join(results_dir, "curves.json")) as f:
        curves = json.load(f)
    with open(os.path.join(results_dir, "results.json")) as f:
        results = json.load(f)

    nets = {
        "attacker_dqn": {"layers": linear_layers(attackers["DQN"].agent.model), "grid": ATTACKER_GRID.round(4).tolist()},
        "keeper_dqn": {"layers": linear_layers(keepers["DQN"].agent.model), "grid": KEEPER_GRID.round(4).tolist()},
        "attacker_ddpg": {"actor": linear_layers(attackers["DDPG"].agent.actor),
                          "critic": linear_layers(attackers["DDPG"].agent.critic),
                          "low": ATTACKER_LOW, "high": ATTACKER_HIGH},
        "keeper_ddpg": {"actor": linear_layers(keepers["DDPG"].agent.actor),
                        "critic": linear_layers(keepers["DDPG"].agent.critic),
                        "low": KEEPER_LOW, "high": KEEPER_HIGH},
    }

    # a few fixed inputs so the JS forward pass can be checked against PyTorch
    tests = []
    for lat, dist in ((0.0, 11.0), (-7.5, 7.0), (6.0, 15.0)):
        tests.append({"role": "attacker", "lat": lat, "dist": dist,
                      "dqn": attackers["DQN"].act(lat, dist).tolist(),
                      "ddpg": attackers["DDPG"].act(lat, dist).tolist()})
    for read, lat, dist in (((1.0, 2.0), 0.0, 11.0), ((5.5, 0.4), 4.0, 16.0)):
        tests.append({"role": "keeper", "read": list(read), "lat": lat, "dist": dist,
                      "dqn": keepers["DQN"].act(np.array(read), lat, dist).tolist(),
                      "ddpg": keepers["DDPG"].act(np.array(read), lat, dist).tolist()})

    smoothed = {}
    for name, log in curves["curves"].items():
        smoothed[name] = {k: smooth(v) for k, v in log.items() if k != "reward"}
        smoothed[name]["reward"] = smooth(log["reward"])
    data = {
        "params": PARAMS,
        "nets": nets,
        "curves": {"episodes": {k: len(v["reward"]) for k, v in curves["curves"].items()},
                   "window": 50, "series": smoothed},
        "matrix": results["matrix"],
        "positions": results["positions"],
        "showcase": {k: {"counts": v["counts"]} for k, v in results["showcase"].items()},
        "n_eval": results["n_shots"],
        "tests": tests,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
