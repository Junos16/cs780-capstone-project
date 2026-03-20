from __future__ import annotations
import os
import random
import json
import numpy as np
import torch

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STATE_SPACE_SIZE = 2**18

class DynaQAgent:
    def __init__(self, n_actions=5):
        self.q_table = np.ones((STATE_SPACE_SIZE, n_actions), dtype=np.float32) * 10.0
        self.model = {}
        self.visited_states = []

def obs_to_state(obs: np.ndarray) -> int:
    return np.sum(2**np.where(obs > 0)[0])

def train(level: int, wall_obstacles: bool, episodes: int, config_file: str = None, render: bool = False):
    print("Training Dyna-Q agent for level", level, "with wall obstacles", wall_obstacles, "for", episodes, "episodes")
    difficulty = 0 if level == 1 else 1 if level == 2 else 2 if level == 3 else 3
    
    # Default hyperparameters
    config = {
        "gamma": 0.99,
        "alpha": 0.1, 
        "planning_steps": 50,     
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay_episodes": int(episodes * 0.8),
        "seed": 42,
        "max_steps": 1000,
        "scaling_factor": 5,
        "arena_size": 500,
        "box_speed": 2
    }
    
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config.update(json.load(f))

    gamma = config["gamma"]
    alpha = config["alpha"]
    planning_steps = config["planning_steps"]
    seed = config["seed"]
    
    random.seed(seed)
    np.random.seed(seed)    
    
    agent = DynaQAgent(n_actions=len(ACTIONS))

    def get_epsilon_greedy_action(stateID: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return random.choice(range(len(ACTIONS)))
        else:
            return int(np.argmax(agent.q_table[stateID]))
    
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
        stateID = obs_to_state(obs)
        epsilon = max(config["eps_end"], config["eps_start"] - episode / config["eps_decay_episodes"])
        
        episode_return = 0.0

        for _ in range(config["max_steps"]):
            action = get_epsilon_greedy_action(stateID, epsilon)
            
            next_obs, reward, done = env.step(ACTIONS[action], render=render)
            next_stateID = obs_to_state(next_obs)
            
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(agent.q_table[next_stateID])
            
            agent.q_table[stateID, action] = agent.q_table[stateID, action] + alpha * (target - agent.q_table[stateID, action])
            
            if stateID not in agent.model:
                agent.model[stateID] = {}
            if action not in agent.model[stateID]:
                agent.visited_states.append((stateID, action)) 

            agent.model[stateID][action] = (reward, next_stateID, done)

            for _ in range(planning_steps):
                s, a = random.choice(agent.visited_states)
                r, s_prime, is_terminal = agent.model[s][a]
                
                if is_terminal:
                    target = r
                else:
                    target = r + gamma * np.max(agent.q_table[s_prime])
                
                agent.q_table[s, a] = agent.q_table[s, a] + alpha * (target - agent.q_table[s, a])
            
            stateID = next_stateID
            episode_return += reward
            
            if done:
                break
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes} return={episode_return:.1f} eps={epsilon:.3f}")
    
    os.makedirs("models", exist_ok=True)
    out_path = f"models/dynaq_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"
    torch.save(torch.from_numpy(agent.q_table), out_path)
    print(f"Saved Q-table to {out_path}")

_Q_TABLE = None

def _load_once(level: int, wall_obstacles: bool):
    global _Q_TABLE
    if _Q_TABLE is None:
        wpath = f"models/dynaq_level{level}{'_wall' if wall_obstacles else ''}_weights.pth"
        _Q_TABLE = torch.load(wpath, map_location="cpu", weights_only=True).numpy()
    return _Q_TABLE
    
def policy(obs: np.ndarray, rng: np.random.Generator, level: int=1, wall_obstacles: bool=False) -> str:
    _load_once(level, wall_obstacles)
    stateID = obs_to_state(obs)
    best_action_idx = int(np.argmax(_Q_TABLE[stateID]))
    return ACTIONS[best_action_idx]

def get_optuna_params(trial, total_episodes):
    params = {}
    params["gamma"] = trial.suggest_float("gamma", 0.9, 1.0)
    params["alpha"] = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    params["planning_steps"] = trial.suggest_categorical("planning_steps", [10, 50, 100, 250])
    eps_fraction = trial.suggest_float("eps_decay_fraction", 0.4, 0.9)
    params["eps_decay_episodes"] = int(total_episodes * eps_fraction)
    return params