from random import randint, shuffle

with open("answer.csv", "r") as f:
    answer = f.readlines()
    answer = [x[:-1].split(",") for x in answer]

for i in range(len(answer)):
    ans = answer[i]
    r = randint(0, len(ans))
    for i in range(r):
        x = randint(0, len(ans)-1)
        y = randint(0, len(ans)-1)
        ans[x], ans[y] = ans[y], ans[x]


with open("predict.csv", "w") as f:
    for ans in answer:
        f.write(f"{','.join(ans)}\n")
