import os
import random
import numpy as np
from tqdm import tqdm
random.seed(0)
np.random.seed(0)


def load_data(dataset_path, split="::"):
    data_item_list = []
    # data views
    # userId::movieId::rating::timestamp (ml-1m)
    for data_item in open(dataset_path):
        # data_item_list:[(6040, 858, 4, 956703932), (1, 593, 3, 1112484661), ...]
        temp_tuple = list(data_item.strip().split(split)[:4])
        temp_tuple[0] = int(temp_tuple[0])-1  # userID
        temp_tuple[1] = int(temp_tuple[1])-1  # movieID
        temp_tuple[2] = int(temp_tuple[2])  # rating
        temp_tuple[3] = int(temp_tuple[3])  # timestamp
        data_item_list.append(tuple(temp_tuple))
    return data_item_list


def get_ui_matrix(data):
    # build U-I matrix
    user_list = [i[0] for i in data]
    item_list = [i[1] for i in data]
    ui_matrix = np.zeros((max(user_list) + 1, max(item_list) + 1))

    for each_interaction in tqdm(data, total=len(data), ascii=True):
        """ ui_matrix[userID][movieID] = rating """
        ui_matrix[each_interaction[0]
                  ][each_interaction[1]] = each_interaction[2]

    return ui_matrix


class MFModel():
    def __init__(self, R, K, lr=0.0002, beta=0.02, steps=5000):
        """
            R: user-item matrix
            K: latent features
            lr: learning rate
            beta: regularization parameter
            steps: iterations
        """
        # data distribution
        self.user_num, self.item_num = R.shape
        # save params
        self.R = R
        self.K = K
        self.lr = lr
        self.beta = beta
        self.steps = steps

    def init(self):
        # init LFM
        self.W = np.random.rand(self.user_num, self.K)
        self.H = np.random.rand(self.item_num, self.K)
        # 把稀疏矩陣存成陣列
        self.samples = []
        for ui in tqdm(range(self.user_num), ascii=True):
            for uj in tqdm(range(self.user_num), leave=bool(ui == self.user_num-1), ascii=True):
                for i in tqdm(range(self.item_num), leave=bool(uj == self.user_num-1), ascii=True):
                    for j in tqdm(range(self.item_num), leave=bool(i == self.item_num-1), ascii=True):
                        if ui != uj or i == j:
                            continue
                        if self.R[ui, i] > self.R[uj, j]:
                            self.samples.append((int(ui), int(i), int(j)))

    def train(self):
        # do SGD [steps] round
        history = []
        process = tqdm(range(self.steps), total=self.steps, ascii=True)
        for i in process:
            np.random.shuffle(self.samples)
            self.updateBPR()
            error = self.rmse()
            history.append((i, error))
            if (i == 0) or ((i+1) % (self.steps / 10) == 0):
                process.set_description(
                    f"Iteration: {i+1}; error = {error:.4}")
        return history

    def updateBPR(self):
        """
            x_uij = W[u] dot H[i] - W[u] dot H[j]
            W[u][f] = W[u][f] - learning_rate * ( exp_x / (1 + exp_x) * ( H[i][f] - H[j][f] ) + beta * W[u][f] )
            H[i][k] = H[i][k] - learning_rate * ( exp_x / (1 + exp_x) * ( W[u][f] ) + beta * H[i][f] )
            H[j][k] = H[j]][k] - learning_rate * ( exp_x / (1 + exp_x) * ( -W[u]][f] ) + beta * H[i][f] )
        """
        for u, i, j in self.samples:
            x_uij = self.W[u, :].dot(self.H[i, :].T) - \
                self.W[u, :].dot(self.H[j, :].T)
            exp_x = np.exp(-x_uij)
            partial_BPR = exp_x / (1 + exp_x)
            # 透過SGD更新參數
            self.W[u, :] -= self.lr * \
                (partial_BPR * (self.H[i, :] -
                 self.H[j, :]) + self.beta * self.W[u, :])
            self.H[i, :] -= self.lr * \
                (partial_BPR * self.W[u, :] + self.beta * self.H[i, :])
            self.H[j, :] -= self.lr * \
                (partial_BPR * (-self.W[u, :]) + self.beta * self.H[i, :])

    def full_matrix(self):
        return self.W.dot(self.H.T)

    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def mae(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = []
        for x, y in zip(xs, ys):
            error.append(abs(self.R[x, y] - predicted[x, y]))
        return np.average(error)

    def rmse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = []
        for x, y in zip(xs, ys):
            error.append(pow(self.R[x, y] - predicted[x, y], 2))
        return np.sqrt(np.average(error))


if __name__ == "__main__":
    ratingfile_path = os.path.join("../data/ml-1m", "ratings.dat")
    origin_data = load_data(ratingfile_path)
    ui_matrix = get_ui_matrix(origin_data)

    mf = MFModel(ui_matrix, K=100, lr=0.00001, beta=0.3, steps=1)
    mf.init()
    mf.train()

    # user_list = [i[0] for i in obs_dataset]
    # for each_user in tqdm(list(set(user_list)), total=len(list(set(user_list)))):
    #     user_ratings = mf.full_matrix()[each_user].tolist()
    #     topN = [(i, user_ratings.index(i)) for i in user_ratings]
    #     # sort Top N
    #     topN = [i[1] for i in sorted(topN, key=lambda x:x[0], reverse=True)][:10]
    #     print("------ each_user ------")
    #     print(each_user)
    #     print("------ temp_topN ------")
    #     print(temp_topN)
