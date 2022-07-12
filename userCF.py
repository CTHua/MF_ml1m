import os
import random
import numpy as np
import pandas as pd
random.seed(0)
PRINT_STEP = 1


def cosineSimilarity(a, b):
    if np.linalg.norm(a)*np.linalg.norm(b) == 0:
        return np.nan
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


class UserBasedCF():
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

    def buildUserSim(self):
        for i in range(self.userNum):
            for j in range(self.userNum):
                if i*self.userNum+j != 0 and i*self.userNum+j % PRINT_STEP == 0:
                    print(
                        f"{'buildUserSim':15}: Finish {i*self.userNum+j}, Total={self.userNum*self.userNum}")
                self.userSimMatrix[i][j] = cosineSimilarity(
                    self.ratingMatrix[i], self.ratingMatrix[j])

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
    model = UserBasedCF()
    model.loadfile(trainSet)
    model.buildUserSim()
    # model.recommend()
