from ARS_TD3_v2 import *
import pickle

env = 'HalfCheetah-v2'
seeds = [1, 2, 3, 4, 5]

for seed in seeds:
    hp = HP(env_name=env, seed=seed, num_samples=10,hidden_size=300, syn_step=1, noise=0.01,
            learning_rate=0.01)
    agent = ARS_TD3(hp)
    reward, step = agent.train()
    pickle.dump((reward, step), open(env + '_ars_seeds_' + str(seed), mode='wb'))
