from math import sqrt, cos, sin, exp

def norm(vec):
    return sqrt(sum([x_i**2 for x_i in vec]))

def getDerivarive(func, x, ind, h=10**(-5)):
    n = len(x)
    return (func([x[j] + h * (ind == j) for j in range(n)]) - func([x[j] - h * (ind == j) for j in range(n)])) / (2 * h)

def getGrad(func, x, h=10**(-5)):
    n = len(x)
    result = [0] * n
    for i in range(n):
        result[i] = getDerivarive(func, x, i)
    return result
