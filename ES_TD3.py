import numpy as np
import tensorflow as tf
import gym
import utils
import math
import collections
import datetime
import td3_network


# from td3_network import *


class HP:
    def __init__(self, env_name='Hopper-v2', total_episodes=1000, learning_steps=1000, gpu=0, update_time=1,
                 episode_length=1000, total_steps=int(1e6), lr=1e-3, action_bound=1, num_samples=10, noise=0.02,
                 std_dev=0.03, batch_size=100, elite_percentage=0.2, mutate=0.9, crossover=0.2, hidden_size=64, seed=1):
        self.env = gym.make(env_name)
        np.random.seed(seed)
        self.env.seed(seed)
        tf.set_random_seed(seed)
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.shape[0]
        self.total_episodes = total_episodes
        self.episode_length = episode_length
        self.total_steps = total_steps
        self.update_time = update_time
        self.lr = lr
        self.action_bound = action_bound
        self.num_samples = num_samples
        self.noise = noise
        self.stddev = std_dev
        self.batch_size = batch_size
        self.elite_percentage = elite_percentage
        self.mutate = mutate
        self.crossover = crossover
        self.hidden_size = hidden_size
        self.normalizer = utils.Normalizer(self.input_size)
        self.batch_size = batch_size
        # config = tf.ConfigProto(device_count={'GPU': gpu})
        self.learning_steps = learning_steps
        self.td3_agent = td3_network.TD3(self.input_size, self.output_size, 1, namescope=str(seed),hidden_size=hidden_size)


class Policy:
    def __init__(self, hp):
        self.hp = hp
        # 针对每个层
        self.w1 = np.random.randn(self.hp.input_size, self.hp.hidden_size) * self.hp.stddev
        self.b1 = np.zeros([self.hp.hidden_size, ])
        self.w2 = np.random.randn(self.hp.hidden_size, self.hp.hidden_size) * self.hp.stddev
        self.b2 = np.zeros([self.hp.hidden_size, ])
        self.w3 = np.random.randn(self.hp.hidden_size, self.hp.output_size) * self.hp.stddev
        self.b3 = np.zeros([self.hp.output_size, ])
        self.bc = None

    def get_params(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def set_params(self, param_list):
        self.w1 = param_list[0]
        self.b1 = param_list[1]
        self.w2 = param_list[2]
        self.b2 = param_list[3]
        self.w3 = param_list[4]
        self.b3 = param_list[5]

    def get_action(self, state, delta=None):
        state = np.reshape(state, [1, self.hp.input_size])
        if delta is None:
            output1 = np.maximum(np.dot(state, self.w1) + self.b1, 0)
            output2 = np.maximum(np.dot(output1, self.w2) + self.b2, 0)
            action = np.tanh(np.reshape(np.dot(output2, self.w3) + self.b3, [self.hp.output_size, ]))
        else:
            output1 = np.maximum(np.dot(state, self.w1 + delta[0]) + self.b1 + delta[1], 0)
            output2 = np.maximum(np.dot(output1, self.w2 + delta[2]) + self.b2 + delta[3], 0)
            action = self.hp.action_bound * np.tanh(
                np.reshape(np.dot(output2, self.w3 + delta[4]) + self.b3 + delta[5], [self.hp.output_size, ]))
        return action * self.hp.action_bound

    def evaluate(self, delta=None):
        # 根据当前state执在环境中执行一次，返回获得的reward和novelty
        # env为环境，为了防止多次初始化这里传入环境
        total_reward = 0
        num_steps = 0
        obs = self.hp.env.reset()
        for i in range(self.hp.episode_length):
            self.hp.normalizer.observe(obs)
            # action = np.clip(self.get_action(self.hp.normalizer.normalize(obs), delta=delta), -1, 1)
            action = np.clip(self.get_action(obs, delta=delta), -1, 1)
            next_obs, reward, done, _ = self.hp.env.step(action)
            if i == self.hp.episode_length:
                done = True
            self.hp.td3_agent.store(obs, next_obs, action, reward, done)
            obs = next_obs
            total_reward += reward
            num_steps += 1
            if done:
                break
        return total_reward, num_steps

    def mutate(self):
        # 随机选择一个参数进行突变。按照ERL的来。
        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05
        param_list = self.get_params().copy()
        ssne_probabilities = np.random.uniform(0, 1, len(param_list))  # 先不乘2
        new_param_list = param_list.copy()
        for i, params in enumerate(param_list):  # 针对每个参数
            if np.random.random() < ssne_probabilities[i] and i % 2 == 0:  # 找几个点突变
                parameter_shape = params.shape
                current_params = params.flatten()
                test = int(current_params.size * num_mutation_frac)  # 有可能为0
                num_mutations = np.random.randint(test)
                for _ in range(num_mutations):
                    mutation_index = np.random.randint(len(current_params))
                    random_num = np.random.random()
                    if random_num < super_mut_prob:
                        current_params[mutation_index] += np.random.randn() * super_mut_strength * current_params[
                            mutation_index]
                    elif random_num < reset_prob:
                        current_params[mutation_index] = np.random.randn()
                    else:
                        current_params[mutation_index] += np.random.randn() * mut_strength * current_params[
                            mutation_index]
                current_params = np.reshape(current_params, parameter_shape)
                new_param_list[i] = current_params
        self.set_params(new_param_list)


class Population:
    def __init__(self, hp):
        # 创建n个population
        self.pop = collections.deque(maxlen=hp.num_samples)
        for i in range(hp.num_samples):
            self.pop.append(Policy(hp))

    def eval_fitness(self):
        total_steps = 0
        fitness = []
        for policy in self.pop:
            reward, step = policy.evaluate()
            total_steps += step
            fitness.append(reward)
        return fitness, total_steps

    def cross_over(self, index1, index2):
        # 对population进行修改,根据index进行杂交
        param1 = self.pop[index1].get_params().copy()
        param2 = self.pop[index2].get_params().copy()
        out_param1 = param1.copy()
        out_param2 = param2.copy()
        for i in range(len(param1)):
            current_shape = param1[i].shape
            parameter_size = param1[i].size
            num_cross_overs = np.random.randint(parameter_size)
            c_param1 = param1[i].flatten()
            c_param2 = param2[i].flatten()
            for j in range(num_cross_overs):
                receiver_choice = np.random.random()
                cross_over_index = np.random.randint(parameter_size)
                if receiver_choice < 0.5:
                    c_param2[cross_over_index] = c_param1[cross_over_index]
                else:
                    c_param2[cross_over_index] = c_param1[cross_over_index]
            c_param1 = np.reshape(c_param1, current_shape)
            c_param2 = np.reshape(c_param2, current_shape)
            out_param1[i] = c_param1
            out_param2[i] = c_param2
        self.pop[index1].set_params(out_param1)
        self.pop[index2].set_params(out_param2)

    def mutate(self, index):
        # 从index中选
        policy = self.pop[index]
        policy.mutate()
        self.pop[index] = policy

    def update_policy(self, policy, index):
        self.pop[index] = policy


class ERL_TD3:
    def __init__(self, hp):
        self.hp = hp

    def train(self):
        population = Population(self.hp)
        total_step = 0
        total_step_list = []
        total_reward = []
        for i in range(self.hp.total_episodes):
            start = datetime.datetime.now()
            fitness, steps = population.eval_fitness()
            total_step += steps
            sorted_index = np.argsort(fitness)

            other_index = sorted_index[:-int(len(sorted_index) * self.hp.elite_percentage)]
            total_reward.append(fitness[sorted_index[-1]])
            for index1 in range(len(sorted_index)):
                for index2 in other_index:
                    population.cross_over(index1, index2)
            # mutate_index = np.random.choice(len(other_index), 2, replace=False)
            # population.mutate(mutate_index)  # 这个地方有问题，并不是所有的突变，而是按照突变的选择两个突变。
            for index in other_index:
                if np.random.random() < self.hp.mutate:
                    population.mutate(index)
            env = self.hp.env
            obs = env.reset()
            td3_reward = 0
            for step in range(self.hp.episode_length):  # 再按照TD3采集一次样本
                action = self.hp.td3_agent.get_action(np.reshape(obs, (1, self.hp.input_size)))
                action = (action + np.random.normal(0, 0.1, size=action.shape)).clip(
                    env.action_space.low,
                    env.action_space.high)
                action = np.reshape(action, [-1])
                next_obs, reward, done, _ = env.step(action)
                td3_reward += reward
                # self.hp.replay_buffer.add((self.hp.normalizer.normalize(obs), self.hp.normalizer.normalize(next_obs),
                #                          action, reward, done))
                self.hp.td3_agent.store(obs, next_obs, action, reward, done)
                obs = next_obs
                if done:
                    #if i > 20:
                    #    self.hp.td3_agent.train(self.hp.learning_steps)
                    break
            if i > 20:
                self.hp.td3_agent.train(self.hp.learning_steps)
            if i % self.hp.update_time is 0 and i is not 1:
                weakest = population.pop[sorted_index[0]]
                weakest.set_params(self.hp.td3_agent.get_params())
                population.update_policy(weakest, sorted_index[0])
            total_step_list.append(total_step + step)
            print('#####')
            print('Episode ', i, ' reward:', fitness[sorted_index[-1]])  # 最好的结果
            print('Running steps:', total_step + step)
            print('Running time:', (datetime.datetime.now() - start).seconds)
            print('TD3 reward is:', td3_reward)

        return total_reward, total_step_list
