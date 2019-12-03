from td3_network import *
from simhash import HashingBonusEvaluator
import pickle

env_name = 'Hopper-v2'
add_bonus = False
if add_bonus:
    save_name = env_name + '_cba_'

env = gym.make(env_name)
agent = TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], action_bound=1)
simhash = HashingBonusEvaluator(dim_key=32,
                                obs_processed_flat_dim=env.observation_space.shape[0] + env.action_space.shape[0])
env.seed(1)
np.random.seed(1)
reward_list = []
for i in range(2000):
    steps = 0
    total_reward = 0
    obs = env.reset()
    gamma=0.99
    for j in range(1000):
        action = agent.get_action(obs.flatten())
        action = (action + np.random.normal(0, 0.2, size=action.shape)).clip(-1, 1)
        state_action_pair = np.concatenate([np.reshape(obs, [-1, 1]), np.reshape(action, [-1, 1])], axis=0)
        next_obs, reward, done, _ = env.step(action)
        simhash.inc_hash(np.reshape(state_action_pair, [1, -1]))
        bonus = simhash.predict_v2(np.reshape(state_action_pair, [1, -1]))
        if add_bonus:
            agent.store(obs, next_obs, action, reward + bonus, done)
        else:
            agent.store(obs, next_obs, action, reward, done)
        steps += 1
        total_reward += gamma**j*reward
        obs = next_obs
        if done:
            test_reward = 0
            obs = env.reset()
            for step in range(1000):
                action = agent.get_action_target(obs.flatten())
                next_obs, reward, done, _ = env.step(action)
                test_reward += gamma**step*reward
                obs = next_obs
                if done:
                    print('Episdoe', i, ' Reward is: ', test_reward)
                    reward_list.append(test_reward)
                    break
            agent.train(steps)
            break

pickle.dump(reward_list, open(save_name + 'td3', mode='wb'))
