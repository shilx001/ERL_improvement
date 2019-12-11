from ARS_DDPG import *
import pickle

env = 'Hopper-v2'
seeds = [3, 4, 5]

for seed in seeds:
    hp = HP(env_name=env, seed=seed, weight=0.01,hidden_size=100,syn_step=20,learning_steps=500)
    agent = ARS_DDPG(hp)
    reward, step = agent.train()
    pickle.dump((reward, step), open(env + '_arsddpg_seeds_' + str(seed), mode='wb'))
