from ES_TD3_CBA import *
import pickle

env_name = 'Hopper-v2'
seed = [1, 2, 3, 4, 5]

for i in seed:
    hp = HP(env_name=env_name, seed=i, total_episodes=1000, beta=1, hidden_size=300, namescope=str(i))
    agent = ERL_TD3(hp)
    reward, step = agent.train()
    pickle.dump((reward, step), open(env_name + '_es_td3_seeds_' + str(i), 'wb'))
