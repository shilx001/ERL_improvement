from ARS import *
import pickle

env = 'Hopper-v2'
seeds = [3, 4, 5]

for seed in seeds:
    hp = HP(env_name=env, seed=seed,num_samples=8,hidden_size=300)
    agent = ARS_TD3(hp)
    reward, step = agent.train()
    pickle.dump((reward, step), open(env + '_ars_seeds_' + str(seed), mode='wb'))
