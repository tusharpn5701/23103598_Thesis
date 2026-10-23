"""
Simulated penalty-area environment shared by every agent in the project (v2).

The thesis used three slightly different environments (attacker phases 1-3,
goalkeeper, combined). v2 keeps the same geometry (7.32 m x 2.44 m goal,
shots from inside the box) but puts everything in one place so that training
and the head-to-head simulation use exactly the same rules.

Coordinates
-----------
Goal plane:  u in [0, GOAL_W] left post -> right post,  v in [0, GOAL_H] ground -> bar
Pitch:       shooter at (lat, dist): lateral offset from the goal centre and
             distance from the goal line, both in metres.

What changed from the thesis environments
-----------------------------------------
* The shot is no longer perfect: the ball lands at aim + Gaussian error whose
  size grows with distance and with the angle to goal. Aiming at the corner
  is now a real trade-off (harder to save, easier to miss).
* The goalkeeper can no longer teleport. It starts in the middle of the goal
  and can only travel as far as the ball's flight time allows
  (reaction time + dive speed), so long shots are easier to save than close ones.
* The goalkeeper sees a noisy read of where the ball is going, instead of the
  attacker's exact target.
* The shooter position actually matters, so the attacker state (lat, dist) is
  informative instead of being ignored by the reward.
All constants live in PARAMS and are exported for the website so the browser
simulation runs the identical model.
"""
from __future__ import annotations

import numpy as np

PARAMS = {
    "goal_w": 7.32,
    "goal_h": 2.44,
    "lat_range": [-10.0, 10.0],      # shooter lateral offset from goal centre (m)
    "dist_range": [6.0, 16.5],       # shooter distance from goal line (m) - inside the box
    "ball_speed": 26.0,              # m/s, a firm placed shot
    "shot_noise_base": 0.10,         # m, execution error at 0 m
    "shot_noise_per_m": 0.025,       # extra m of error per metre of range
    "keeper_start": [3.66, 1.0],     # keeper starts centred, body centre ~1 m high
    "keeper_reaction": 0.18,         # s before the keeper can move
    "keeper_speed": 5.0,             # m/s dive speed
    "keeper_read_noise": 0.35,       # m, error in the keeper's read of the shot
    "save_radius": 1.0,              # ball within 1 m of keeper's hands = save (thesis 'deflect' radius)
    "catch_radius": 0.5,             # thesis 'catch' radius (used for reward only)
}

GOAL_W = PARAMS["goal_w"]
GOAL_H = PARAMS["goal_h"]
KEEPER_START = np.array(PARAMS["keeper_start"], dtype=np.float64)

GOAL, SAVED, MISSED = 0, 1, 2
OUTCOME_NAMES = {GOAL: "goal", SAVED: "saved", MISSED: "missed"}


# --------------------------------------------------------------------------- geometry
def shot_range(lat: float, dist: float, u: float = GOAL_W / 2) -> float:
    """Straight-line distance from the shooter to a point on the goal line."""
    return float(np.hypot(lat - (u - GOAL_W / 2), dist))


def shot_noise(lat: float, dist: float) -> tuple[float, float]:
    """Std of the landing error (horizontal, vertical) for a shot from (lat, dist)."""
    r = shot_range(lat, dist)
    sigma = PARAMS["shot_noise_base"] + PARAMS["shot_noise_per_m"] * r
    cos_angle = dist / r                     # 1 when straight on, smaller from wide
    return sigma / cos_angle, sigma


def keeper_reach(lat: float, dist: float) -> float:
    """How far (m) the keeper can travel before the ball arrives."""
    t = shot_range(lat, dist) / PARAMS["ball_speed"]
    return max(0.0, t - PARAMS["keeper_reaction"]) * PARAMS["keeper_speed"]


def keeper_final_position(target, lat: float, dist: float) -> np.ndarray:
    """Clip the keeper's intended position to what is reachable in time."""
    target = np.clip(np.asarray(target, dtype=np.float64), [0.0, 0.0], [GOAL_W, GOAL_H])
    move = target - KEEPER_START
    n = np.linalg.norm(move)
    reach = keeper_reach(lat, dist)
    if n > reach:
        move = move * (reach / (n + 1e-9))
    return KEEPER_START + move


def is_on_target(u: float, v: float) -> bool:
    return 0.0 <= u <= GOAL_W and 0.0 <= v <= GOAL_H


# --------------------------------------------------------------------------- state encoders
def attacker_obs(lat: float, dist: float) -> np.ndarray:
    """Normalised attacker state: roughly in [-1, 1]."""
    lo, hi = PARAMS["dist_range"]
    return np.array([lat / 10.0, (dist - (lo + hi) / 2) / ((hi - lo) / 2)], dtype=np.float32)


def keeper_obs(read_u: float, read_v: float, lat: float, dist: float) -> np.ndarray:
    """Normalised keeper state: noisy read of the shot + where it was struck from.

    The thesis keeper state was [ball_x, ball_y, gk_x, gk_y]; the keeper start is
    constant so v2 swaps it for the shooter position, which decides how much time
    the keeper has.
    """
    a = attacker_obs(lat, dist)
    return np.array([read_u / GOAL_W * 2 - 1, read_v / GOAL_H * 2 - 1, a[0], a[1]], dtype=np.float32)


# --------------------------------------------------------------------------- environment
class PenaltyEnv:
    """One shot = one step = one episode-step. Used by every training script.

    reset()            -> samples a shooter position
    shoot(aim)         -> ball landing point (aim + execution noise)
    keeper_read(ball)  -> noisy observation of the landing point
    resolve(ball, gk)  -> outcome (GOAL / SAVED / MISSED) and the keeper's final position
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.lat = 0.0
        self.dist = 11.0

    def reset(self, lat: float | None = None, dist: float | None = None):
        self.lat = self.rng.uniform(*PARAMS["lat_range"]) if lat is None else float(lat)
        self.dist = self.rng.uniform(*PARAMS["dist_range"]) if dist is None else float(dist)
        return self.lat, self.dist

    def shoot(self, aim) -> np.ndarray:
        su, sv = shot_noise(self.lat, self.dist)
        u = aim[0] + self.rng.normal(0, su)
        v = max(0.0, aim[1] + self.rng.normal(0, sv))   # ball cannot go below the ground
        return np.array([u, v])

    def keeper_read(self, ball) -> np.ndarray:
        n = PARAMS["keeper_read_noise"]
        return np.asarray(ball) + self.rng.normal(0, n, size=2)

    def resolve(self, ball, keeper_target):
        if keeper_target is None:                       # empty goal
            gk = np.array([np.nan, np.nan])
            return (GOAL if is_on_target(*ball) else MISSED), gk
        gk = keeper_final_position(keeper_target, self.lat, self.dist)
        if not is_on_target(*ball):
            return MISSED, gk
        if np.linalg.norm(gk - ball) < PARAMS["save_radius"]:
            return SAVED, gk
        return GOAL, gk


# --------------------------------------------------------------------------- rewards
def attacker_reward(outcome: int) -> float:
    """Outcome reward from the thesis' combined environment (+10 goal, -10 otherwise)."""
    return 10.0 if outcome == GOAL else -10.0


def keeper_reward(outcome: int, ball, gk) -> float:
    """+10 save / -10 goal, plus the thesis' distance-to-ball shaping term (0..5)
    and +2 for a clean catch (ball within the thesis' 0.5 m catch radius).

    A shot that misses the target gives 0: the keeper's action did not matter.
    """
    if outcome == MISSED:
        return 0.0
    d = float(np.linalg.norm(np.asarray(gk) - np.asarray(ball)))
    d_max = float(np.hypot(GOAL_W, GOAL_H))
    shaping = (1.0 - d / d_max) * 5.0
    catch_bonus = 2.0 if d < PARAMS["catch_radius"] else 0.0   # clean catch, as in the thesis
    return (10.0 if outcome == SAVED else -10.0) + shaping + catch_bonus


# --------------------------------------------------------------------------- scripted opponents
class SimplisticAttacker:
    """Baseline from the thesis: aims uniformly at random inside the goal."""
    name = "Simplistic"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, lat, dist, explore=False):
        return np.array([self.rng.uniform(0, GOAL_W), self.rng.uniform(0, GOAL_H)])


class HeuristicKeeper:
    """Scripted keeper: dives straight at its read of the ball."""
    name = "Heuristic"

    def act(self, read, lat, dist, explore=False):
        return np.clip(read, [0, 0], [GOAL_W, GOAL_H])
