from random import shuffle
N = 100  # the total number of application members
L = 30  # predict length
predict = list(range(1, L+1))
predict = list(map(lambda x: str(x), predict))

with open("answer.csv", "w") as f:
    for i in range(N):
        shuffle(predict)
        f.write(f"{','.join(predict)}\n")
