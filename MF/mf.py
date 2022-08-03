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

    def train(self):
        # init LFM
        self.P = np.random.rand(self.user_num, self.K)
        self.Q = np.random.rand(self.item_num, self.K)
        # param when use biasSVD
        self.b_u = np.zeros(self.user_num)
        self.b_i = np.zeros(self.item_num)
        self.miu = np.mean(self.R[np.where(self.R != 0)])
        # 把稀疏矩陣存成陣列
        self.samples = []
        for i in range(self.user_num):
            for j in range(self.item_num):
                if self.R[i, j] > 0:
                    self.samples.append((i, j, self.R[i, j]))
        # do SGD [steps] round
        history = []
        process = tqdm(range(self.steps), total=self.steps, ascii=True)
        for i in process:
            np.random.shuffle(self.samples)
            self.SVD()
            error = self.rmse()
            history.append((i, error))
            if (i == 0) or ((i+1) % (self.steps / 10) == 0):
                process.set_description(
                    f"Iteration: {i+1}; error = {error:.4}")
        return history

    def SVD(self):
        """
            err[u][i] = r[u][i] - r_hat[u][i]
            r_hat[u][i] = P[u] dot Q[i]
            P[u][k] = P[u][k] + learning_rate * ( r[u][i] - r_hat[u][i] * Q[i][k] - beta * P[u][k] )
            Q[i][k] = Q[i][k] + learning_rate * ( r[u][i] - r_hat[u][i] * P[u][k] - beta * Q[i][k] )
        """
        for u, i, r in self.samples:
            r_hat = self.P[u, :].dot(self.Q[i, :].T)
            e = r - r_hat
            # 透過SGD更新參數
            tempP = self.lr * (e * self.Q[i, :] - self.beta * self.P[u, :])
            self.Q[i, :] += self.lr * \
                (e * self.P[u, :] - self.beta * self.Q[i, :])
            self.P[u, :] += tempP

    def biasSVD(self):
        """
            b_u := b_u + learning_rate * ( r[u][i] - r_hat[u][i] - beta * b_u )
            b_i := b_i + learning_rate * ( r[u][i] - r_hat[u][i] - beta * b_i )
            err[u][i] = r[u][i] - r_hat[u][i]
            r_hat[u][i] = miu(average of R) + b_u + b_i + P[u] dot Q[i]
            P[u][k] = P[u][k] + learning_rate * ( r[u][i] - r_hat[u][i] * Q[i][k] - beta * P[u][k] )
            Q[i][k] = Q[i][k] + learning_rate * ( r[u][i] - r_hat[u][i] * P[u][k] - beta * Q[i][k] )
        """
        for u, i, r in self.samples:
            # error
            r_hat = self.miu + self.b_u[u] + \
                self.b_i[i]+self.P[u, :].dot(self.Q[i, :].T)
            e = r - r_hat
            # 透過SGD更新參數
            self.b_u[u] += self.lr * (e - self.beta * self.b_u[u])
            self.b_i[i] += self.lr * (e - self.beta * self.b_i[i])
            self.P[u, :] += self.lr * \
                (e * self.Q[i, :] - self.beta * self.P[u, :])
            self.Q[i, :] += self.lr * \
                (e * self.P[u, :] - self.beta * self.Q[i, :])

    def full_matrix(self):
        return self.miu + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)

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
