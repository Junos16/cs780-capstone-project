from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOActorCritic(nn.Module):
    def __init__(self, in_dim=36, n_actions=5):
        super().__init__()
        # Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
    def get_value(self, x):
        return self.critic(x)

_model: Optional[PPOActorCritic] = None
_prev_obs: Optional[np.ndarray] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_MAX_REPEAT = 2
_CLOSE_Q_DELTA = 0.05

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m = PPOActorCritic()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=False)
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _prev_obs, _last_action, _repeat_count
    _load_once()
    
    if _prev_obs is None or np.sum(np.abs(obs - _prev_obs)) > 5:
        delta_obs = np.zeros_like(obs, dtype=np.float32)
    else:
        delta_obs = (obs - _prev_obs).astype(np.float32)
    
    _prev_obs = obs.copy()
    
    s = np.concatenate([obs, delta_obs]).astype(np.float32)
    x = torch.from_numpy(s).unsqueeze(0)

    logits = _model.actor(x).squeeze(0).numpy()
    best = int(np.argmax(logits))

    if _last_action is not None:
        order = np.argsort(-logits)
        best_l, second_l = float(logits[order[0]]), float(logits[order[1]])
        if (best_l - second_l) < _CLOSE_Q_DELTA:
            if _repeat_count < _MAX_REPEAT:
                best = _last_action
                _repeat_count += 1
            else:
                _repeat_count = 0
        else:
            _repeat_count = 0

    _last_action = best
    return ACTIONS[best]
