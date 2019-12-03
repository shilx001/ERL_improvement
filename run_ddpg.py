import numpy as np
import gym
from ddpg import *



env = gym.make('InvertedPendulum-v2')
agent = DDPG(a_dim=env.action_space.shape[0], s_dim=env.observation_space.shape[0], a_bound=1)

for episode in range(10000):
    total_reward = 0
    obs = env.reset()
    for step in range(env.spec.timestep_limit):
        action = agent.choose_action(obs)
        action = np.clip(action + np.random.randn() * 0.2, -1, 1)
        next_obs, reward, done, _ = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)
        agent.learn()
        obs = next_obs
        total_reward += reward
        if done:
            print('Episode ', episode, 'step:',step,' reward is:', total_reward)
            break
