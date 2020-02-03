import argparse
import datetime
import gym
import itertools
import math
import mxnet
import random

from sac import SAC
from sac.utils import MemoryBuffer
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Soft Actor-Critic (SAC) in MXNet")
parser.add_argument(
    "--env-name",
    default="MountainCarContinuous-v0",
    help="Gym environment (default: MountainCarContinuous-v0)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="discount factor for reward (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    help="target smoothing coefficient(τ) (default: 0.005)",
)
parser.add_argument(
    "--lr", type=float, default=0.0003, help="learning rate (default: 0.0003)"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.2,
    help="Relative importance of the entropy term against the reward (default: 0.2)",
)
parser.add_argument(
    "--automatic_entropy_tuning",
    type=bool,
    default=False,
    help="Automatically adjust α (default: False)",
)
parser.add_argument(
    "--seed", type=int, default=42, help="random seed (default: 123456)"
)
parser.add_argument(
    "--batch_size", type=int, default=256, help="batch size (default: 256)"
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=1000,
    help="maximum number of episodes (default: 1e3)",
)
parser.add_argument(
    "--hidden_size", type=int, default=64, help="hidden size (default: 64)"
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=1,
    help="number of updates betweeen actions (default: 1)",
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.3,
    help="ε-greedy exploration factor (default: 0.3)",
)
parser.add_argument(
    "--start_steps",
    type=int,
    default=1e4,
    help="Steps to enforce random actions (default: 1e4)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1,
    help="Value target update per no. of updates per step (default: 1)",
)
parser.add_argument(
    "--replay_size", type=int, default=1e6, help="size of replay buffer (default: 1e6)",
)
parser.add_argument(
    "--eval_X",
    type=int,
    default=10,
    help="Evaluates a policy a policy every X episodes (default: 10; -1 to disable it)",
)
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Set verbosity  [0: disabled, 1: every `eval_X` episodes, 2: every episode] (default: 1)",
)
parser.add_argument("--gpu", action="store_true", help="run on GPU (default: False)")
args = parser.parse_args()

# Environment
env = gym.make(args.env_name)

# Seed tools
mxnet.random.seed(args.seed)
random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC.load_model("./saved_models/2020-02-02_22-22-27_SAC_MountainCarContinuous-v0.pkl")

# Training Loop
total_numsteps = 0
updates = 0
avg_reward = 0.0

for _ in range(args.eval_X):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, eval=True)
        env.render()

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        state = next_state
    avg_reward += episode_reward
avg_reward /= args.eval_X

writer.add_scalar("avg_reward/test", avg_reward, i_episode)

print("----------------------------------------")
print(
    "Avg. Reward: {}".format(
        round(avg_reward, 2)
    )
)
print("----------------------------------------")
env.close()

agent.save_model("saved_models/{}".format(filename))
