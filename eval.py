import gym
import mxnet as mx

from sac import SAC


agent = SAC.load_model("./saved_models/2020-02-02_22-22-27_SAC_MountainCarContinuous-v0.pkl")
agent.set_context(mx.cpu())

mx.Context.default_ctx = mx.Context(mx.gpu(), 0)

env = gym.make("MountainCarContinuous-v0")

# Training Loop
total_numsteps = 0
updates = 0
avg_reward = 0.0
eval_X = 10

for _ in range(eval_X):
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
avg_reward /= eval_X

print("----------------------------------------")
print(
    "Avg. Reward: {}".format(
        round(avg_reward, 2)
    )
)
print("----------------------------------------")
env.close()
