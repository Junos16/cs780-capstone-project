import argparse
import os
import numpy as np
import cv2

from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# Key mapping matches manual_play.py exactly:
#   q = L22 (small left turn)
#   a = L45 (large left turn)  / left arrow
#   w = FW  (forward)          / up arrow
#   e = R22 (small right turn)
#   d = R45 (large right turn) / right arrow
#   r = reset episode
#   ESC = quit and save


def main():
    parser = argparse.ArgumentParser(
        description="Record human demonstrations for behavioral cloning."
    )
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/bc_demos.npz")
    parser.add_argument("--wall", action="store_true")
    args = parser.parse_args()

    difficulty = {1: 0, 2: 2, 3: 3}[args.level]

    all_obs, all_actions = [], []

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=5,
            arena_size=500,
            max_steps=1000,
            wall_obstacles=args.wall,
            difficulty=difficulty,
            seed=ep * 42,
        )
        obs = env.reset(seed=ep * 42)
        env.render_frame()

        print(
            f"Episode {ep+1}/{args.episodes}. "
            f"Controls: a/left=L45, q=L22, w/up=FW, e=R22, d/right=R45, r=reset, ESC=quit"
        )

        ep_obs, ep_actions = [], []
        quit_all = False

        while True:
            key = cv2.waitKey(0) & 0xFF

            action_str = None
            if key == ord("a") or key == 81:    # a or left arrow
                action_str = "L45"
            elif key == ord("q"):
                action_str = "L22"
            elif key == ord("w") or key == 82:  # w or up arrow
                action_str = "FW"
            elif key == ord("e"):
                action_str = "R22"
            elif key == ord("d") or key == 83:  # d or right arrow
                action_str = "R45"
            elif key == ord("r"):
                break
            elif key == 27:                     # ESC
                quit_all = True
                break

            if action_str is None:
                continue

            action_idx = ACTIONS.index(action_str)
            ep_obs.append(obs.copy())
            ep_actions.append(action_idx)

            obs, reward, done = env.step(action_str, render=True)
            print(f"  step {len(ep_obs)}  reward={reward:.1f}", end="\r")

            if done:
                print(f"\n  Episode done.")
                break

        all_obs.extend(ep_obs)
        all_actions.extend(ep_actions)
        print(f"  Recorded {len(ep_obs)} steps. Total so far: {len(all_obs)}")

        if quit_all:
            break

    cv2.destroyAllWindows()

    if not all_obs:
        print("No transitions recorded.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        observations=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int32),
    )
    print(f"Saved {len(all_obs)} transitions to {args.output}")


if __name__ == "__main__":
    main()
