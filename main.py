import argparse
import datetime
import gym
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
    "--lr", type=float, default=0.003, help="learning rate (default: 0.0003)"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Relative importance of the entropy term against the reward (default: 0.5)",
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
    default=100,
    help="number of updates betweeen actions (default: 1)",
)
parser.add_argument(
    "--env_steps",
    type=int,
    default=2500,
    help="Maximum number of steps for each episode"
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.8,
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
    "--render",
    type=int,
    default=0,
    help="Render mode [0: disabled, 1: every episode, 2: every evaluation] (default: 0)",
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
env._max_episode_steps = int(args.env_steps)

# Seed tools
mxnet.random.seed(args.seed)
random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, **vars(args))

# TensorboardX
filename = "{}_SAC_{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name
)
writer = SummaryWriter(logdir="logs/{}".format(filename))

# Memory buffer
memory = MemoryBuffer(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in range(args.num_episodes):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    transitions = list()

    while not done:
        p = random.random()
        threshold = args.epsilon + math.exp(-total_numsteps / args.start_steps)
        if p < threshold:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if args.render > 1:
            env.render()

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon  mask = (1-d)
        mask = 1 if episode_steps == env._max_episode_steps else int(not done)
        transitions.append((state, action, reward, next_state, mask))

        state = next_state

    memory.push_bulk(transitions)  # Append transition to memory
    if len(memory) > args.batch_size:
        # Number of updates per step in environment
        for i in range(args.updates_per_step):
            # Update parameters of all the networks
            (
                critic_1_loss,
                critic_2_loss,
                policy_loss,
                ent_loss,
                alpha,
            ) = agent.update_parameters(memory, args.batch_size, updates)

            writer.add_scalar("loss/critic_1", critic_1_loss, updates)
            writer.add_scalar("loss/critic_2", critic_2_loss, updates)
            writer.add_scalar("loss/policy", policy_loss, updates)
            writer.add_scalar("loss/entropy_loss", ent_loss, updates)
            writer.add_scalar("entropy_temprature/alpha", alpha, updates)
            updates += 1

    writer.add_scalar("reward/train", episode_reward, i_episode)
    if args.verbose == 2:
        print(
            "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
                i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
            )
        )

    if args.eval_X > 0 and args.verbose > 0:
        if i_episode % args.eval_X == 0 and i_episode > 0:
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)
                if args.render > 0:
                    env.render()

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state

            writer.add_scalar("avg_reward/test", episode_reward, i_episode)

            print("----------------------------------------")
            print(
                "Test Episodes: {}, Avg. Reward: {}".format(
                    i_episode, round(episode_reward, 2)
                )
            )
            print("----------------------------------------")
            env.close()

agent.save_model("saved_models/{}".format(filename))
