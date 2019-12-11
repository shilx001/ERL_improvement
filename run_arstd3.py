from ARS_TD3 import *
import pickle

env = 'Hopper-v2'
seeds = [1,2, 3, 4, 5]

for seed in seeds:
    hp = HP(env_name=env, seed=seed, weight=0.01,hidden_size=300,syn_step=10)
    agent = ARS_TD3(hp)
    reward, step = agent.train()
    pickle.dump((reward, step), open(env + '_ars_seeds_' + str(seed), mode='wb'))
