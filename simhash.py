import numpy as np


class HashingBonusEvaluator(object):
    """Hash-based count bonus for exploration.
    Tang, H., Houthooft, R., Foote, D., Stooke, A., Chen, X., Duan, Y., Schulman, J., De Turck, F., and Abbeel, P. (2017).
    #Exploration: A study of count-based exploration for deep reinforcement learning.
    In Advances in Neural Information Processing Systems (NIPS)
    """

    def __init__(self, dim_key=128, obs_processed_flat_dim=None, bucket_sizes=None, beta=1):
        # Hashing function: SimHash
        # dim_key: key的维度
        # obs_processed_flat_dim: 观察到的state拉直后的维度
        # bucket_sizes可以先不管
        if bucket_sizes is None:
            # Large prime numbers
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        mods_list = []
        for bucket_size in bucket_sizes:  # 针对每个bucket值
            mod = 1
            mods = []
            for _ in range(dim_key):  # 针对每个维度,吧mod加入到mods来，然后mode就除以二
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)  # [1,6]
        self.mods_list = np.asarray(mods_list).T  # [128,6]矩阵，存的是每个维度的mod值
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))  # [6, 999983],全0矩阵，很有可能是存state的count值
        self.projection_matrix = np.random.normal(size=(obs_processed_flat_dim, dim_key))  # [6, 128] 投影矩阵，就是初始化的那个函数
        self.beta = beta

    def compute_keys(self, obss):
        # 根据输入的值转换为对应的哈希key
        binaries = np.sign(np.asarray(obss).dot(self.projection_matrix))  # [1,128], 乘上投影矩阵，并变成-1,0,1三个离散化值
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes  # 值转化为int后
        return keys

    def inc_hash(self, obss):  # 就是统计tables值，在相应Observation上面加1
        keys = self.compute_keys(obss)  # 得出key值
        for idx in range(len(self.bucket_sizes)):  # 针对每个index
            np.add.at(self.tables[idx], keys[:, idx], 1)  # 在keys的位置加1

    def query_hash(self, obss):  # 查询state哈希值的次数,state必须是[1,obs_processed_flat_dim]列
        keys = self.compute_keys(obss)  #
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def fit_before_process_samples(self, obs):  # 处理observations，使之存储在表里，再查询
        if len(obs.shape) == 1:
            obss = [obs]
        else:
            obss = obs
        before_counts = self.query_hash(obss)  # 这个貌似没用到
        self.inc_hash(obss)

    def predict(self, obs):  # 看看有几个值，如果sqrt(counts)小于1，则按1计算
        counts = self.query_hash(obs)
        return 1. / np.maximum(1., np.sqrt(counts))

    def predict_v2(self, obs):
        counts = self.query_hash(obs)
        return self.beta / np.sqrt(counts)
