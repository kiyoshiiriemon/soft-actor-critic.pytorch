import time
import os
import glob
import argparse
import numpy as np
import torch
import gym
from gym import wrappers
import pybullet_envs

from model import GaussianPolicy
from utils import grad_false


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='AntBulletEnv-v0')
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.log_name:
        log_dir = os.path.join('logs', args.env_id, args.log_name)
    else:
        env_dir = os.path.join('logs', args.env_id, '*')
        dirs = glob.glob(env_dir)
        log_dir = max(dirs, key=os.path.getctime)
        print(f'using {log_dir}')

    env = gym.make(args.env_id)
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    policy = GaussianPolicy(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        hidden_units=[256, 256]).to(device)

    policy.load(os.path.join(log_dir, 'model', 'policy.pth'))
    grad_false(policy)

    def exploit(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, action = policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    env.render()
    while True:
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            env.render()
            action = exploit(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        print(f'total reward: {episode_reward}')
        time.sleep(1)


if __name__ == '__main__':
    run()
