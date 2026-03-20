from __future__ import annotations
import os
import random
import json
import numpy as np
import torch

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_FEATURES = 18 

class LinUCBAgent:
    def __init__(self, n_actions=5, n_features=18):
        self.n_actions = n_actions
        self.n_features = n_features
        
        self.A = np.zeros((n_actions, n_features, n_features), dtype=np.float32)
        for a in range(n_actions):
            self.A[a] = np.eye(n_features, dtype=np.float32)
            
        self.b = np.zeros((n_actions, n_features, 1), dtype=np.float32)

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None, render: bool = False):
    print("Training LinUCB agent for level", level, "with wall obstacles", wall_obstacles, "for", episodes, "episodes")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3
    
    # Default hyperparameters
    config = {
        "alpha": 1.5, 
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2
    }
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config.update(json.load(f))

    alpha = config["alpha"]
    seed = config["seed"]
    
    random.seed(seed)
    np.random.seed(seed)    
    
    # Bias feature added to make it so that expected_r isn't just all 0 when it passes through [0, 0, ..., 0]
    agent = LinUCBAgent(n_actions=len(ACTIONS), n_features=N_FEATURES + 1)
    
    for episode in range(episodes):
        env = OBELIX(
            scaling_factor=config["scaling_factor"],
            arena_size=config["arena_size"],
            max_steps=config["max_steps"],
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=config["box_speed"],
            seed=seed+episode
        )

        obs = env.reset(seed=seed+episode)
        episode_return = 0.0

        for _ in range(config["max_steps"]):
            x_t = np.append(obs, 1).reshape(-1, 1).astype(np.float32)
            
            p = np.zeros(len(ACTIONS))

            for a in range(len(ACTIONS)):
                A_inv = np.linalg.inv(agent.A[a])
                theta_a = A_inv @ agent.b[a]
                expected_r = (theta_a.T @ x_t).item()
                explore = alpha * np.sqrt((x_t.T @ A_inv @ x_t).item())
                p[a] = expected_r + explore
            
            action = np.argmax(p)
            next_obs, reward, done = env.step(ACTIONS[action], render=render)
            
            agent.A[action] = agent.A[action] + (x_t @ x_t.T)
            agent.b[action] = agent.b[action] + (reward * x_t) 

            obs = next_obs
            episode_return += reward
            
            if done:
                break
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes} return={episode_return:.1f}")
    
    os.makedirs("models", exist_ok=True)
    out_path = f"models/linucb_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"
    
    state_dict = {
        "A": torch.from_numpy(agent.A),
        "b": torch.from_numpy(agent.b)
    }
    torch.save(state_dict, out_path)
    print(f"Saved LinUCB matrices to {out_path}")

_LINUCB_STATE = None

def _load_once(level: int, wall_obstacles: bool):
    global _LINUCB_STATE
    if _LINUCB_STATE is None:
        wpath = f"models/linucb_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"
        loaded = torch.load(wpath, map_location="cpu", weights_only=True)
        _LINUCB_STATE = {
            "A": loaded["A"].numpy(),
            "b": loaded["b"].numpy()
        }
    return _LINUCB_STATE
    
def policy(obs: np.ndarray, rng: np.random.Generator, level: int=1, wall_obstacles: bool=False) -> str:
    state = _load_once(level, wall_obstacles)
    
    x_t = np.append(obs, 1.0).reshape(-1, 1).astype(np.float32)
    expected_rewards = np.zeros(len(ACTIONS))

    for a in range(len(ACTIONS)):
        A_inv = np.linalg.inv(state["A"][a])
        theta_a = A_inv @ state["b"][a]
        expected_rewards[a] = (theta_a.T @ x_t).item()
    
    best_action_idx = int(np.argmax(expected_rewards))
    return ACTIONS[best_action_idx]

def get_optuna_params(trial, total_episodes):
    params = {}
    params["alpha"] = trial.suggest_float("alpha", 0.05, 5.0, log=True)
    return params