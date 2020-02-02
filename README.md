# Soft Actor-critic in MXNet

This repository is an implementation of the paper

> Tuomas Haarnoja and Aurick Zhou and Pieter Abbeel and Sergey Levine (2018). [*Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*](https://arxiv.org/abs/1801.01290).CoRR, abs/1801.01290.

This code was implemented in MXNet 1.4.1. The authors also have an [implementation](https://github.com/haarnoja/sac) using tensorflow. There's also a comprehensive [implementation](https://github.com/pranz24/pytorch-soft-actor-critic) of SAC in pytorch


## How to use
```
>>> python main.py --help

Soft Actor-Critic (SAC) in MXNet

optional arguments:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Gym environment (default: MountainCarContinuous-v0)
  --eval EVAL           Evaluates a policy a policy every X episodes
                        (default: 10; -1 to disable it)
  --gamma GAMMA         discount factor for reward (default: 0.99)
  --tau TAU             target smoothing coefficient(τ) (default: 0.005)
  --lr LR               learning rate (default: 0.0003)
  --alpha ALPHA         Relative importance of the entropy term against the
                        reward (default: 0.2)
  --automatic_entropy_tuning AUTOMATIC_ENTROPY_TUNING
                        Automatically adjust α (default: False)
  --seed SEED           random seed (default: 123456)
  --batch_size BATCH_SIZE
                        batch size (default: 256)
  --num_steps NUM_STEPS
                        maximum number of steps (default: 1000000)
  --hidden_size HIDDEN_SIZE
                        hidden size (default: 64)
  --updates_per_step UPDATES_PER_STEP
                        model updates per simulator step (default: 1)
  --start_steps START_STEPS
                        Steps sampling random actions (default: 10000)
  --target_update_interval TARGET_UPDATE_INTERVAL
                        Value target update per no. of updates per step
                        (default: 1)
  --replay_size REPLAY_SIZE
                        size of replay buffer (default: 10000000)
  --gpu                 run on GPU (default: False)
```


