import numpy as np


class AliasMethod:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.alias = np.zeros(len(probabilities), dtype=int)

        # init alias table to -1, mean no index
        for i in range(len(probabilities)):
            self.alias[i] = -1
        self.prob = np.zeros(len(probabilities), dtype=float)

        # Step 1: Normalize probabilities
        probabilities = probabilities / np.sum(probabilities)

        # Step 2: Divide probabilities into two groups
        small = []
        large = []
        for i, prob in enumerate(probabilities):
            self.prob[i] = len(probabilities) * prob
            if self.prob[i] < 1.0:
                small.append(i)
            else:
                large.append(i)

        # Step 3: Assign probabilities to each group
        while small and large:
            small_index = small.pop()
            large_index = large.pop()
            self.alias[small_index] = large_index
            self.prob[large_index] = self.prob[large_index] - \
                (1.0 - self.prob[small_index])
            if self.prob[large_index] < 1.0:
                small.append(large_index)
            else:
                large.append(large_index)

        while large:
            self.prob[large.pop()] = 1.0

        while small:
            self.prob[small.pop()] = 1.0

    def sample(self):
        i = np.random.randint(0, len(self.probabilities))
        if np.random.random() < self.prob[i]:
            return i
        else:
            return self.alias[i]

    def print(self):
        print("probabilities: ", self.probabilities)
        print("prob: ", self.prob)
        print("alias: ", self.alias)


if __name__ == '__main__':
    probabilities = [0.5, 0.3, 0.1, 0.1]
    alias_method = AliasMethod(probabilities)
    alias_method.print()
    print(alias_method.sample())
