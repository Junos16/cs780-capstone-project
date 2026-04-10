"""
Double DQN with n-step bootstrapped returns for OBELIX — phase 4.

Extends ddqn.py by replacing 1-step TD targets with n-step returns.

Motivation:
  The push reward (+500) arrives at episode termination, which can be up to
  1000 steps away. Standard 1-step Bellman backup propagates this reward
  one step per update — requiring ~500 updates just for the signal to reach
  the states where the agent first contacts the box. n-step returns
  propagate it ~n× faster by including n actual rewards in the target.

n-step target:
  y = Σ_{i=0}^{n-1} γ^i r_{t+i}  +  γ^n Q_target(s_{t+n}, argmax_a Q_online(s_{t+n}, a))
  The first term is the actual discounted return over n steps; the second
  bootstraps from the target net n steps into the future.

NStepBuffer mechanics:
  A deque of maxlen=n accumulates (s, a, r, s', done) tuples. Once full (or
  when done=True), the oldest transition is popped with its n-step return
  computed via the deque. On episode end, the buffer is flushed: remaining
  transitions get partial returns from however many steps remain.

Tradeoff:
  Larger n → faster reward propagation but higher bias (bootstrap from a
  less-converged Q-function n steps ahead). n=8 is a practical middle
  ground: push signal propagates 8× faster while bootstrap error remains
  bounded.

Everything else (DDQN target, Huber loss, replay buffer, ε-greedy) is
identical to ddqn.py.
Config adds: n_steps (default 8).
"""
from __future__ import annotations
import os
import json
import math
import collections
import random
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


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s.copy(), a, r, s2.copy(), done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(np.array(a), dtype=torch.long),
            torch.tensor(np.array(r), dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(np.array(d), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buf = collections.deque(maxlen=n)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))
        if len(self.buf) < self.n and not done:
            return None
        G = 0.0
        final_s2 = s2
        final_done = done
        actual_n = 0
        for i, (si, ai, ri, s2i, di) in enumerate(self.buf):
            G += self.gamma**i * ri
            actual_n = i + 1
            final_s2 = s2i
            final_done = di
            if di:
                break
        s0, a0 = self.buf[0][0], self.buf[0][1]
        return s0, a0, G, final_s2, final_done, actual_n

    def clear(self):
        self.buf.clear()


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
    "n_steps": 8,
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
    replay = ReplayBuffer(cfg["replay_size"])
    nstep_buf = NStepBuffer(cfg["n_steps"], cfg["gamma"])
    gamma_n = cfg["gamma"] ** cfg["n_steps"]

    eps_start = cfg["eps_start"]
    eps_end = cfg["eps_end"]
    eps_decay_frac = cfg["eps_decay_frac"]
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
        nstep_buf.clear()

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

            result = nstep_buf.push(feat, act_idx, reward, feat2, float(done))
            if result is not None:
                s0, a0, G, sn, dn, actual_n = result
                replay.push(s0, a0, G, sn, dn)

            ep_reward += reward
            total_steps += 1
            last_fw = last_fw2
            obs = obs2

            if len(replay) >= cfg["warmup_steps"]:
                s_t, a_t, r_t, s2_t, d_t = replay.sample(cfg["batch_size"])

                with torch.no_grad():
                    best_actions = qnet(s2_t).argmax(dim=1, keepdim=True)
                    q_next = target_net(s2_t).gather(1, best_actions).squeeze(1)
                    target_q = r_t + gamma_n * q_next * (1.0 - d_t)

                pred_q = qnet(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                loss = F.huber_loss(pred_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % cfg["target_update"] == 0:
                    target_net.load_state_dict(qnet.state_dict())

            if done:
                break

        # Flush remaining transitions in n-step buffer at episode end
        remaining = list(nstep_buf.buf)
        for start_idx in range(len(remaining)):
            G = 0.0
            final_s2 = remaining[-1][3]
            final_done = remaining[-1][4]
            for i, (si, ai, ri, s2i, di) in enumerate(remaining[start_idx:]):
                G += cfg["gamma"]**i * ri
                final_s2 = s2i
                final_done = di
                if di:
                    break
            s0, a0 = remaining[start_idx][0], remaining[start_idx][1]
            replay.push(s0, a0, G, final_s2, final_done)
        nstep_buf.clear()

        ep_rewards.append(ep_reward)
        rolling_mean = float(np.mean(ep_rewards[-10:]))
        eps_now = max(eps_end, eps_start - (eps_start - eps_end) * total_steps /
                      (episodes * cfg["max_steps"] * eps_decay_frac))
        print(f"Ep {ep+1}/{episodes}  reward={ep_reward:.1f}  "
              f"rolling10={rolling_mean:.1f}  eps={eps_now:.3f}  steps={total_steps}")

        if trial is not None:
            trial.report(rolling_mean, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

    os.makedirs("models", exist_ok=True)
    base_name = prefix if prefix else f"ddqn_nstep_level{level}{'_wall' if wall_obstacles else ''}"
    out_path = f"models/{base_name}_weights.pth"
    torch.save(qnet.state_dict(), out_path)
    print(f"Saved model to {out_path}")


def _load_once():
    global _MODEL
    if _MODEL is None:
        base_name = (_CURRENT_PREFIX if _CURRENT_PREFIX
                     else f"ddqn_nstep_level{_CURRENT_LEVEL}{'_wall' if _CURRENT_WALL else ''}")
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
    params["n_steps"] = trial.suggest_categorical("n_steps", [5, 8, 12])
    return params
