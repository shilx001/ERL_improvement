from ES_DDPG import *
import pickle

env_name = 'Hopper-v2'
seed = [1, 2, 3, 4, 5]

for i in seed:
    hp = HP(env_name=env_name, seed=i, hidden_size=300, total_episodes=1000, learning_steps=100, namescope=str(i))
    agent = ERL_TD3(hp)
    reward, step = agent.train()
    pickle.dump((reward,step), open(env_name + '_es_ddpg_seeds_' + str(i), 'wb'))
