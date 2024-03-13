import argparse
import gymnasium as gym
import numpy as np
from itertools import count
from collections import deque
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.core.random import categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make('CartPole-v1')
env.reset(seed=args.seed)
eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        
        self.affine1 = nn.Linear(in_dims, 128)
        # self.dropout = nn.Dropout(p=0.5)
        self.ln = nn.LayerNorm(dims=1, affine=True)
        self.affine2 = nn.Linear(128, out_dims)

    def __call__(self, x):
        x = self.affine1(x)
        # x = self.dropout(x)
        x = self.ln(x)
        x = nn.leaky_relu(x)
        action_scores = self.affine2(x)
        return mx.softmax(action_scores, axis=0)

policy = Policy(4, 2)
mx.eval(policy.parameters())

num_episodes = 500
lr_schedule = optim.linear_schedule(1e-3, 1e-5, 400)
optimizer = optim.Adam(learning_rate=lr_schedule)
state_ = [policy.state, optimizer.state]


def run_episode():
    
    saved_log_probs = []
    rewards = []
    actions = []
    state, _ = env.reset()
    ep_reward = 0
    
    for _ in range(1, 10000):
        state = mx.array(state)
        probs = policy(state)
        action = np.random.choice(2, p=np.array(probs))
        
        saved_log_probs.append(probs.log())
        actions.append(action)
        
        state, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        ep_reward += reward
        if done:
            break
        
    R = 0
    returns = deque()
    for r in rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = list(returns)
    returns = mx.array(returns)
    returns = (returns - returns.mean()) / (returns.var()**0.5 + eps)

    saved_log_probs = mx.array(saved_log_probs)
    actions = mx.array(actions)
    policy_loss = nn.losses.cross_entropy(saved_log_probs, actions)
    policy_loss = (policy_loss * returns).mean()

    return policy_loss, ep_reward

grad_fn = nn.value_and_grad(policy, run_episode)

running_reward = 10
for i_episode in range(1, num_episodes):
    
    t0 = time.time()
    (loss, ep_reward), grads = grad_fn()
    optimizer.update(policy, grads)
    mx.eval(policy.state, optimizer.state)
    t1 = time.time()
    total_time = t1-t0
        
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f} \t LR: {:.10f} \t Time {:.5f}'.format(
              i_episode, ep_reward, running_reward, optimizer.learning_rate.item(), total_time))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, ep_reward))
        break

