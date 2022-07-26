N = 100  # the total number of application members
L = 30   # predict length
K = 5    # at k


class Fraction():
    def __init__(self, num, deno):
        self.num = num
        self.deno = deno
        self.reduce()

    def reduce(self):
        if self.num == 0:
            return
        gcd = Fraction.GCD(self.num, self.deno)
        self.num //= gcd
        self.deno //= gcd

    def __add__(self, other):
        lcm = self.deno//Fraction.GCD(self.deno, other.deno)*other.deno
        num = (self.num*(lcm//self.deno)+other.num*(lcm//other.deno))
        deno = lcm
        return Fraction(num, deno)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self, other):
        if type(other) == int:
            return Fraction(self.num*other, self.deno)

        deno = self.deno*other.deno
        num = self.num*other.num
        return Fraction(num, deno)

    def __str__(self):
        return f"({self.num}/{self.deno})"

    def __repr__(self):
        return self.__str__()

    def __float__(self):
        return self.num/self.deno

    @staticmethod
    def GCD(x, y):
        if y == 0:
            return x
        return Fraction.GCD(y, x % y)


def sumFraction(fracList):
    return sum(fracList)


def calcAP(ans, pred):
    count = 0
    precision = []
    for i in range(K):
        if ans[i] == pred[i]:
            count += 1
            precision.append(Fraction(count, i+1))
    if len(precision) == 0:
        return Fraction(0, 1)
    sumPrecision = sum(precision)
    sumPrecision = Fraction(1, count)*sumPrecision

    return sumPrecision


def calcMAP(ansList, predList):
    apList = []
    L = len(ansList)
    for i in range(L):
        ans = ansList[i]
        pred = predList[i]
        apList.append(calcAP(ans, pred))

    return (sum(apList)*Fraction(1, L))


if __name__ == "__main__":
    with open("answer.csv", "r") as f:
        answer = f.readlines()
        answer = [x[:-1].split(",") for x in answer]

    with open("predict.csv", "r") as f:
        predict = f.readlines()
        predict = [x[:-1].split(",") for x in predict]

    # print(calcAP(ans=[1, 1, 1, 1, 1], pred=[1, 0, 1, 0, 1]))
    # print(calcMAP(ansList=[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        #   predList=[[1, 1, 1, 0, 0], [1, 0, 1, 0, 1]]))
    mapScore = calcMAP(ansList=answer, predList=predict)
    print(mapScore)
    print(float(mapScore))
