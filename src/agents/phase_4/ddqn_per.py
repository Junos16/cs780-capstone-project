"""
Double DQN + Prioritized Experience Replay for OBELIX — phase 4.

Extends ddqn.py with Prioritized Experience Replay (Schaul et al. 2016).

Motivation:
  In OBELIX, contact events (IR activation, enable_push becoming True) and
  successful pushes occur in a small fraction of all timesteps. A uniform
  replay buffer therefore undersamples these critical transitions. PER
  resamples proportionally to |TD error| so rare, high-information
  transitions resurface more frequently.

PER mechanics:
  - Priority p_i = |δ_i|^α where δ_i is the TD error for transition i.
  - Sampling probability P(i) = p_i / Σ_j p_j.
  - α=0.6: interpolates between uniform (α=0) and pure-priority (α=1).
    Keeps some randomness to avoid always replaying the same few transitions.

Importance-sampling (IS) correction:
  Non-uniform sampling introduces bias: transitions with low priority are
  underrepresented. IS weights w_i = (1 / (N * P(i)))^β correct for this.
  β is annealed from β_start=0.4 → 1.0 over training:
    - Early training: β=0.4, accepts more bias (network is unstable anyway).
    - Late training:  β=1.0, fully corrected gradients for convergence.

Implementation: uses fixed numpy arrays (not deque) for O(1) priority
updates. Priorities are stored per-slot and the maximum priority is tracked
to assign max priority to new transitions (ensures new data is replayed at
least once before being deprioritised).

Everything else (DDQN targets, Huber loss, target net sync) is identical
to ddqn.py.
"""
from __future__ import annotations
import os
import json
import math
import random
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5

def _extract_features(obs, ts_seen, ts_stuck, last_fw):
    f = np.zeros(26, dtype=np.float32)
    f[0:18] = obs.astype(np.float32)
    f[18] = float(np.sum(obs[4:12]))
    f[19] = float(np.sum(obs[0:4]))
    f[20] = float(np.sum(obs[12:16]))
    f[21] = float(obs[16])
    f[22] = float(obs[17])
    f[23] = min(1.0, ts_seen / 50.0)
    f[24] = min(1.0, ts_stuck / 20.0)
    f[25] = last_fw
    return f

def _shaped_reward(raw, obs):
    r = raw
    if obs[17] > 0:
        r += 195.0
    if float(np.sum(obs[:17])) == 0.0 and obs[17] == 0:
        r -= 2.0
    if obs[16] > 0:
        r += 3.0
    return r

def _priv_shaping(env, prev_dist):
    curr_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                          (env.bot_center_y - env.box_center_y)**2)
    r = 0.0
    if not env.enable_push:
        r += 2.0 * (prev_dist - curr_dist)
        dx = env.box_center_x - env.bot_center_x
        dy = env.box_center_y - env.bot_center_y
        if abs(dx) + abs(dy) > 1e-3:
            angle_to_box = math.degrees(math.atan2(dy, dx)) % 360
            diff = abs(angle_to_box - env.facing_angle % 360)
            if diff > 180: diff = 360 - diff
            if diff < 45: r += 0.5
    else:
        r += 1.0
    return r, curr_dist


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, 26), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 26), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def push(self, s, a, r, s2, done):
        max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0
        self.states[self.pos] = s
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.next_states[self.pos] = s2
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        prios = self.priorities[:self.size]
        probs = prios / prios.sum()
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        s = torch.tensor(self.states[indices], dtype=torch.float32)
        a = torch.tensor(self.actions[indices], dtype=torch.long)
        r = torch.tensor(self.rewards[indices], dtype=torch.float32)
        s2 = torch.tensor(self.next_states[indices], dtype=torch.float32)
        d = torch.tensor(self.dones[indices], dtype=torch.float32)
        w = torch.tensor(weights, dtype=torch.float32)

        return s, a, r, s2, d, w, indices

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = (np.abs(td_errors) + 1e-6) ** self.alpha

    def __len__(self):
        return self.size


config = {
    "lr": 1e-4,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_frac": 0.5,
    "batch_size": 64,
    "replay_size": 50000,
    "target_update": 1000,
    "warmup_steps": 1000,
    "seed": 42,
    "max_steps": 1000,
    "scaling_factor": 5,
    "arena_size": 500,
    "box_speed": 2,
    "use_privileged": True,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
}

# Inference globals
_MODEL: Optional[QNet] = None
_ts_seen = 100
_ts_stuck = 100
_last_action: Optional[int] = None
_repeat = 0

_CURRENT_LEVEL = 1
_CURRENT_WALL = False
_CURRENT_PREFIX: Optional[str] = None


def train(level, wall_obstacles, episodes, config_file=None, render=False, prefix=None, trial=None):
    global _CURRENT_LEVEL, _CURRENT_WALL, _CURRENT_PREFIX

    _CURRENT_LEVEL = level
    _CURRENT_WALL = wall_obstacles
    _CURRENT_PREFIX = prefix

    difficulty = {1: 0, 2: 2, 3: 3}.get(level, 0)

    cfg = dict(config)
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            cfg.update(json.load(f))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    qnet = QNet()
    target_net = QNet()
    target_net.load_state_dict(qnet.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(qnet.parameters(), lr=cfg["lr"])
    replay = PrioritizedReplayBuffer(cfg["replay_size"], alpha=cfg["per_alpha"])

    eps_start = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    eps_decay_frac = cfg["eps_decay_frac"]
    beta_start = cfg["per_beta_start"]
    total_steps = 0
    ep_rewards = []

    for ep in range(episodes):
        ep_seed = cfg["seed"] + ep
        env = OBELIX(
            scaling_factor=cfg["scaling_factor"],
            arena_size=cfg["arena_size"],
            max_steps=cfg["max_steps"],
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=cfg["box_speed"],
            seed=ep_seed,
        )
        obs = env.reset(seed=ep_seed)
        t_ep = time.time()

        ts_seen = 100
        ts_stuck = 100
        last_fw = 0.0
        ep_reward = 0.0

        prev_dist = math.sqrt((env.bot_center_x - env.box_center_x)**2 +
                              (env.bot_center_y - env.box_center_y)**2)

        for step in range(cfg["max_steps"]):
            eps = max(eps_end, eps_start - (eps_start - eps_end) * total_steps /
                      (episodes * cfg["max_steps"] * eps_decay_frac))

            if np.any(obs[:17] > 0):
                ts_seen = 0
            else:
                ts_seen += 1
            if obs[17] > 0:
                ts_stuck = 0
            else:
                ts_stuck += 1

            feat = _extract_features(obs, ts_seen, ts_stuck, last_fw)

            if random.random() < eps:
                act_idx = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    q = qnet(torch.tensor(feat).unsqueeze(0))
                act_idx = int(q.argmax(dim=1).item())

            obs2, raw_reward, done = env.step(ACTIONS[act_idx], render=render)

            reward = _shaped_reward(raw_reward, obs2)
            if cfg["use_privileged"]:
                priv_r, prev_dist = _priv_shaping(env, prev_dist)
                reward += priv_r

            last_fw2 = 1.0 if act_idx == 2 else 0.0
            feat2 = _extract_features(obs2, ts_seen, ts_stuck, last_fw2)

            replay.push(feat, act_idx, reward, feat2, float(done))
            ep_reward += reward
            total_steps += 1
            last_fw = last_fw2
            obs = obs2

            if len(replay) >= cfg["warmup_steps"]:
                beta = min(1.0, beta_start + (1.0 - beta_start) * total_steps /
                           (episodes * cfg["max_steps"]))
                s_t, a_t, r_t, s2_t, d_t, is_weights, indices = replay.sample(cfg["batch_size"], beta)

                with torch.no_grad():
                    best_actions = qnet(s2_t).argmax(dim=1, keepdim=True)
                    q_next = target_net(s2_t).gather(1, best_actions).squeeze(1)
                    target_q = r_t + cfg["gamma"] * q_next * (1.0 - d_t)

                pred_q = qnet(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                td_errors = (target_q - pred_q).detach().cpu().numpy()
                loss = (is_weights * F.huber_loss(pred_q, target_q, reduction='none')).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                replay.update_priorities(indices, td_errors)

                if total_steps % cfg["target_update"] == 0:
                    target_net.load_state_dict(qnet.state_dict())

            if done:
                break

        ep_rewards.append(ep_reward)
        rolling_mean = float(np.mean(ep_rewards[-10:]))
        eps_now = max(eps_end, eps_start - (eps_start - eps_end) * total_steps /
                      (episodes * cfg["max_steps"] * eps_decay_frac))
        print(f"Ep {ep+1}/{episodes}  reward={ep_reward:.1f}  "
              f"rolling10={rolling_mean:.1f}  eps={eps_now:.3f}  steps={total_steps}  ({time.time()-t_ep:.1f}s)")

        if trial is not None:
            trial.report(rolling_mean, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    os.makedirs("models", exist_ok=True)
    base_name = prefix if prefix else f"ddqn_per_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = f"models/{base_name}_weights.pth"
    torch.save(qnet.state_dict(), out_path)
    print(f"Saved model to {out_path}")


def _load_once():
    global _MODEL
    if _MODEL is None:
        base_name = (_CURRENT_PREFIX if _CURRENT_PREFIX
                     else f"ddqn_per_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}")
        wpath = f"models/{base_name}_weights.pth"
        _MODEL = QNet()
        _MODEL.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        _MODEL.eval()


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _ts_seen, _ts_stuck, _last_action, _repeat
    _load_once()

    if np.any(obs[:17] > 0):
        _ts_seen = 0
    else:
        _ts_seen += 1
    if obs[17] > 0:
        _ts_stuck = 0
    else:
        _ts_stuck += 1

    last_fw = 1.0 if _last_action == 2 else 0.0
    feat = _extract_features(obs, _ts_seen, _ts_stuck, last_fw)
    q = _MODEL(torch.tensor(feat).unsqueeze(0)).squeeze(0)

    order = q.argsort(descending=True)
    best = int(order[0].item())

    if _last_action is not None:
        top1 = float(q[order[0]])
        top2 = float(q[order[1]])
        if top1 - top2 < 0.02:
            if _repeat < 3:
                best = _last_action
                _repeat += 1
            else:
                _repeat = 0
        else:
            _repeat = 0

    _last_action = best
    return ACTIONS[best]


def get_optuna_params(trial, total_episodes):
    params = {}
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    params["gamma"] = trial.suggest_categorical("gamma", [0.97, 0.99, 0.999])
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    params["use_privileged"] = trial.suggest_categorical("use_privileged", [True, False])
    params["per_alpha"] = trial.suggest_float("per_alpha", 0.4, 0.8)
    params["per_beta_start"] = trial.suggest_float("per_beta_start", 0.3, 0.6)
    return params
