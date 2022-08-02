import os
import random
import numpy as np
from tqdm import tqdm
random.seed(0)


def loadData(dataset_path, split="::"):
    data_item_list = []
    # data views
    # userId::movieId::rating::timestamp (ml-1m)
    for data_item in open(dataset_path):
        # data_item_list:[(6040, 858, 4, 956703932), (1, 593, 3, 1112484661), ...]
        temp_tuple = list(data_item.strip().split(split)[:4])
        temp_tuple[0] = int(temp_tuple[0])  # userID
        temp_tuple[1] = int(temp_tuple[1])  # movieID
        temp_tuple[2] = int(temp_tuple[2])  # rating
        temp_tuple[3] = int(temp_tuple[3])  # timestamp
        data_item_list.append(tuple(temp_tuple))
    # data_item_list = sorted(data_item_list, key=lambda tup: tup[3])
    # data_item_list = sorted(data_item_list, key=lambda tup: tup[0])
    return data_item_list


def getUIMat(data):
    # build U-I matrix
    user_list = [i[0] for i in data]
    item_list = [i[1] for i in data]
    UI_matrix = np.zeros((max(user_list) + 1, max(item_list) + 1))
    # uimat[u][i] = r
    for each_interaction in tqdm(data, total=len(data)):
        UI_matrix[each_interaction[0]
                  ][each_interaction[1]] = each_interaction[2]
    return UI_matrix


class MFModel():
    def __init__(self, R, K, lr=0.0002, beta=0.02, steps=5000):
        '''
            R: rating matrix
            K: latent features
            lr: learning rate
            beta: regularization parameter
            steps: iterations
        '''
        # data distribution
        self.userNum, self.movieNum = R.shape
        self.maxRating = 5

        #
        self.R = R
        self.K = K
        self.lr = lr
        self.beta = beta
        self.steps = steps

        # result
        self.recMovie = {}

    def train(self):
        # init
        self.P = np.random.rand(self.userNum, self.K)
        self.Q = np.random.rand(self.movieNum, self.K)

        #
        self.b_u = np.zeros(self.userNum)
        self.b_i = np.zeros(self.movieNum)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        #
        self.samples = [(i, j, self.R[i, j]) for i in range(self.userNum)
                        for j in range(self.movieNum) if self.R[i, j] > 0]

        # SVD do [steps] round
        training_process = []
        process = tqdm(range(self.steps), total=self.steps)
        for i in process:
            np.random.shuffle(self.samples)
            self.biasSVD()
            mse = self.mse()
            training_process.append((i, mse))
            if (i == 0) or ((i+1) % (self.steps / 10) == 0):
                process.set_description(f"Iteration: {i+1}; error = {mse:.4}")

        return training_process

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

    def biasSVD(self):
        for i, j, r in self.samples:
            # error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            #
            """

            """
            self.b_u[i] += self.lr * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.lr * (e - self.beta * self.b_i[j])

            # update
            self.P[i, :] += self.lr * \
                (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.lr * \
                (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + \
            self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)


if __name__ == "__main__":
    ratingfilePath = os.path.join("../data/ml-1m", "ratings.dat")
    orgData = loadData(ratingfilePath)
    uiMatrix = getUIMat(orgData)

    mf = MFModel(uiMatrix, K=2, lr=0.1, beta=0.3, steps=100)
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
