"""
Evaluate trained agents (thesis Chapters 5, 6 and 7) and draw the figures.

    python evaluate.py              # needs models/ from train.py

Produces results/results.json and figures/*.png:
  * attacker accuracy on an empty goal and vs the scripted keeper, per position
  * keeper save rate vs the simplistic attacker
  * full head-to-head matrix (every attacker vs every keeper)
  * 100-shot simulation plots in the style of thesis Figures 7.1 and 7.3
All evaluation is greedy: no epsilon, no exploration noise.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

torch.set_num_threads(1)

from soccer_env import (PenaltyEnv, GOAL, SAVED, MISSED, GOAL_W, GOAL_H, PARAMS,
                        SimplisticAttacker, HeuristicKeeper)
from agents import load_trained

POSITIONS = [(lat, dist) for dist in (6.0, 11.0, 16.0) for lat in (-8.0, -4.0, 0.0, 4.0, 8.0)]


def play(attacker, keeper, n, seed, positions=None):
    """Run n shots; returns outcome counts and a per-shot log."""
    env = PenaltyEnv(seed)
    counts = np.zeros(3, dtype=int)
    shots = []
    for i in range(n):
        if positions is None:
            lat, dist = env.reset()
        else:
            lat, dist = env.reset(*positions[i % len(positions)])
        aim = attacker.act(lat, dist, explore=False)
        ball = env.shoot(aim)
        target = None if keeper is None else keeper.act(env.keeper_read(ball), lat, dist, explore=False)
        outcome, gk = env.resolve(ball, target)
        counts[outcome] += 1
        shots.append({"lat": lat, "dist": dist, "aim": list(map(float, aim)),
                      "ball": list(map(float, ball)), "gk": list(map(float, gk)), "outcome": int(outcome)})
    return counts, shots


def rates(counts):
    n = counts.sum()
    on_target = counts[GOAL] + counts[SAVED]
    return {"n": int(n), "goal_pct": 100 * counts[GOAL] / n, "saved_pct": 100 * counts[SAVED] / n,
            "missed_pct": 100 * counts[MISSED] / n,
            "save_pct_on_target": 100 * counts[SAVED] / max(1, on_target)}


def main(models_dir="models", out="results", fig_dir="figures", n=5000, seed=2024):
    os.makedirs(out, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    attackers, keepers = load_trained(models_dir)
    attackers = {"Simplistic": SimplisticAttacker(seed), **attackers}
    keepers = {"Empty goal": None, "Heuristic": HeuristicKeeper(), **keepers}

    results = {"params": PARAMS, "n_shots": n, "matrix": {}, "positions": {}, "showcase": {}}

    print("Head-to-head matrix (goal % / saved % / missed %)")
    for an, a in attackers.items():
        results["matrix"][an] = {}
        for kn, k in keepers.items():
            c, _ = play(a, k, n, seed)
            results["matrix"][an][kn] = rates(c)
            r = results["matrix"][an][kn]
            print(f"  {an:>10} vs {kn:<10}  {r['goal_pct']:5.1f} / {r['saved_pct']:5.1f} / {r['missed_pct']:5.1f}")

    print("Per-position scoring rate vs the heuristic keeper (200 shots each)")
    for an in ("DQN", "DDPG", "Simplistic"):
        rows = []
        for pos in POSITIONS:
            c, _ = play(attackers[an], keepers["Heuristic"], 200, seed, positions=[pos])
            rows.append({"lat": pos[0], "dist": pos[1], **rates(c)})
        results["positions"][an] = rows

    for algo in ("DQN", "DDPG"):
        c, shots = play(attackers[algo], keepers[algo], 100, seed)
        results["showcase"][algo] = {"counts": c.tolist(), "shots": shots}
        draw_showcase(shots, f"{algo} attacker vs {algo} goalkeeper: 100 shots",
                      os.path.join(fig_dir, f"simulation_{algo.lower()}.png"))

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    draw_curves(fig_dir)


def draw_showcase(shots, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {GOAL: "#d62728", SAVED: "#1f77b4", MISSED: "#7f7f7f"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    # top-down view: goal along the top edge
    ax1.set_facecolor("#2e7d32")
    ax1.add_patch(plt.Rectangle((-20.16, 0), 40.32, 16.5, fill=False, ec="white", lw=2))
    ax1.add_patch(plt.Rectangle((-9.16, 0), 18.32, 5.5, fill=False, ec="white", lw=2))
    ax1.plot([-GOAL_W / 2, GOAL_W / 2], [0, 0], color="white", lw=6)
    for s in shots:
        x0, y0 = s["lat"], s["dist"]
        x1 = s["ball"][0] - GOAL_W / 2
        ax1.plot([x0, x1], [y0, 0], color=colors[s["outcome"]], lw=1.2, alpha=0.8)
    ax1.set_xlim(-14, 14); ax1.set_ylim(18, -1.5); ax1.set_aspect("equal"); ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Shot origins (top-down)")
    # goal-face view
    ax2.set_facecolor("#e8f5e9")
    ax2.add_patch(plt.Rectangle((0, 0), GOAL_W, GOAL_H, fill=False, ec="black", lw=3))
    for s in shots:
        ax2.scatter(*s["ball"], color=colors[s["outcome"]], s=30, zorder=3)
        if s["outcome"] != MISSED:
            ax2.scatter(*s["gk"], color="orange", s=12, marker="x", zorder=2)
    ax2.set_xlim(-1.5, GOAL_W + 1.5); ax2.set_ylim(-0.2, GOAL_H + 1.2); ax2.set_aspect("equal")
    ax2.set_title("Where the ball arrived (goal face)")
    n = len(shots); g = sum(s["outcome"] == GOAL for s in shots); sv = sum(s["outcome"] == SAVED for s in shots)
    handles = [plt.Line2D([], [], color=colors[k], lw=3, label=f"{lbl}") for k, lbl in
               ((GOAL, f"Goal ({g})"), (SAVED, f"Saved ({sv})"), (MISSED, f"Missed ({n - g - sv})"))]
    handles.append(plt.Line2D([], [], color="orange", marker="x", ls="", label="Keeper final position"))
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_curves(fig_dir, curves_path="results/curves.json", window=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not os.path.exists(curves_path):
        return
    with open(curves_path) as f:
        curves = json.load(f)["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, role, key, label in ((axes[0], "attacker", "goals", "Goals per 10 shots"),
                                 (axes[1], "keeper", "saved", "Saves per 10 shots")):
        for algo, col in (("dqn", "#1f77b4"), ("ddpg", "#d62728")):
            y = np.array(curves[f"{role}_{algo}"][key], dtype=float)
            w = max(1, min(window, len(y)))
            ma = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(y, color=col, alpha=0.12)
            ax.plot(np.arange(w - 1, len(y)), ma, color=col, lw=2, label=f"{algo.upper()} ({w}-ep avg)")
        ax.set_title(f"{role.capitalize()} learning curve"); ax.set_xlabel("Episode"); ax.set_ylabel(label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "learning_curves.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
