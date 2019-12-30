from ARS_DDPG import *
import pickle

env = 'Hopper-v2'
seeds = [3, 4, 5]

for seed in seeds:
    hp = HP(env_name=env, seed=seed, num_samples=4, weight=0.01,hidden_size=300,syn_step=1,learning_steps=100)
    agent = ARS_DDPG(hp)
    reward, step = agent.train()
    pickle.dump((reward, step), open(env + '_arsddpg_seeds_' + str(seed), mode='wb'))
