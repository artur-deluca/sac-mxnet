import argparse
import datetime
import gym
import numpy as np
import itertools
import mxnet

from sac import SAC, ReplayMemory
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Soft Actor-Critic (SAC) in MXNet")
parser.add_argument(
    "--env-name",
    default="MountainCarContinuous-v0",
    help="Gym environment (default: MountainCarContinuous-v0)",
)
parser.add_argument(
    "--eval",
    type=int,
    default=10,
    help="Evaluates a policy a policy every X episodes (default: 10; -1 to disable it)",
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
    "--num_steps",
    type=int,
    default=1000000,
    help="maximum number of steps (default: 1000000)",
)
parser.add_argument(
    "--hidden_size", type=int, default=64, help="hidden size (default: 64)"
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=1,
    help="model updates per simulator step (default: 1)",
)
parser.add_argument(
    "--start_steps",
    type=int,
    default=10000,
    help="Steps sampling random actions (default: 10000)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1,
    help="Value target update per no. of updates per step (default: 1)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=1000000,
    help="size of replay buffer (default: 10000000)",
)
parser.add_argument("--gpu", action="store_true", help="run on GPU (default: False)")
args = parser.parse_args()

# Environment
env = gym.make(args.env_name)

mxnet.random.seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, **vars(args))

# TensorboardX
filename = "{}_SAC_{}".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name
)
writer = SummaryWriter(logdir="logs/{}".format(filename))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

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

        next_state, reward, done, _ = env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon  mask = (1-d)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(
            state, action, reward, next_state, mask
        )  # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar("reward/train", episode_reward, i_episode)
    print(
        "Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2)
        )
    )

    if args.eval > 0 and i_episode % args.eval == 0:
        avg_reward = 0.0
        episodes = args.eval
        for _ in range(episodes):
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
        avg_reward /= episodes

        writer.add_scalar("avg_reward/test", avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        env.close()

agent.save_model("saved_models/{}".format(filename))
