import os
import random
import numpy as np
import pandas as pd
from numpy.linalg import svd as SVD
random.seed(0)
PRINT_STEP = 10000


class MFModel():
    def __init__(self):
        # data distribution
        self.userNum = 6040
        self.movieNum = 3952
        self.maxRating = 5

        # matrix
        self.ratingMatrix = np.zeros((self.userNum, self.movieNum))
        self.userSimMatrix = np.zeros((self.userNum, self.userNum))
        self.recMatrix = np.zeros((self.userNum, self.movieNum))

        # result
        self.recMovie = {}

        # setting
        self.n_rec_movie = 20

    def loadfile(self, trainSet):
        self.ratingNum = len(trainSet)
        for i in range(self.ratingNum):
            if i != 0 and i % PRINT_STEP == 0:
                print(f"{'loadfile':15}: Finish {i}, Total={self.ratingNum}")
            ratingRow = trainSet.loc[i]
            userID = ratingRow["userID"]
            movieID = ratingRow["movieID"]
            rating = ratingRow["rating"]

            self.ratingMatrix[userID-1][movieID-1] = rating/self.maxRating

    def calcK(self):
        r = 100
        u, s, vt = SVD(self.ratingMatrix)
        ur = u[:, 0:r]
        sr = s[0:r]
        vtr = vt[0:r, :]
        return (ur, sr, vtr)

    def recommend(self):
        self.recMatrix = np.matmul(self.ratingMatrix, self.userSimMatrix)
        print(self.recMatrix.shape)

    def evaluate(self, testSet):
        pass


if __name__ == "__main__":
    ratingfilePath = os.path.join("ml-1m", "ratings.dat")
    # read file
    df = pd.read_csv(ratingfilePath, sep="::", header=None,
                     names=["userID", "movieID", "rating", "timestamp"], engine='python')
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    # split train and test set
    trainingRate = 0.01
    testingRate = 1-trainingRate

    trainSet = df[0:int(len(df)*trainingRate)]
    testSet = df[int(len(df)*trainingRate):len(df)]

    print(f"Train Set size = {len(trainSet)}")
    print(f"TestSet Set size = {len(testSet)}")
    model = MFModel()
    model.loadfile(trainSet)
    print(model.calcK())
    # model.recommend()
