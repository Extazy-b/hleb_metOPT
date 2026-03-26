from myMath import *
from gradDest import projectPoint, makeResult

def getGoodStepByGoldSeq(func, left, right, epsilon=10**(-5)):
    phi = (sqrt(5) - 1) / 2
    
    x1 = right - phi * (right - left)
    x2 = left + phi * (right - left)
    
    f1, f2 = func(x1), func(x2)
    
    while (right - left) > epsilon:
        if f1 < f2:
            right = x2
            x2, f2 = x1, f1
            x1 = right - phi * (right - left)
            f1 = func(x1)
        else:
            left = x1
            x1, f1 = x2, f2
            x2 = left + phi * (right - left)
            f2 = func(x2)
            
    return (left + right) / 2


def getRightBorder(stepFunc, initialStep=1.0, growthFactor=2.0):
    right = initialStep
    baseValue = stepFunc(0)
    rightValue = stepFunc(right)

    if rightValue >= baseValue:
        return right

    prevValue = rightValue
    while True:
        nextRight = right * growthFactor
        nextValue = stepFunc(nextRight)
        if nextValue >= prevValue:
            return nextRight
        right = nextRight
        prevValue = nextValue


def fastGradDest(
    func,
    startX,
    gradNormEps,
    pointIncEps,
    funcIncEps,
    MaxIter,
    lineSearchEps=10 ** (-5),
    initialStep=1.0,
    domain=None,
    return_info=False,
):
    currentX = projectPoint(startX, domain)
    trajectory = [currentX[::]]
    iterations = 0
    lastStep = 0

    while iterations < MaxIter:
        gr = getGrad(func, currentX)
        if norm(gr) < gradNormEps:
            result = makeResult(currentX, iterations, trajectory, lastStep, "grad_norm")
            return result if return_info else result["x"]

        currentValue = func(currentX)
        stepFunc = lambda step: func(
            projectPoint(
                [currentX[i] - step * gr[i] for i in range(len(currentX))],
                domain,
            )
        )

        rightBorder = getRightBorder(stepFunc, initialStep=initialStep)
        step = getGoodStepByGoldSeq(stepFunc, 0, rightBorder, epsilon=lineSearchEps)
        newX = projectPoint(
            [currentX[i] - step * gr[i] for i in range(len(currentX))],
            domain,
        )
        newValue = func(newX)

        pointShift = norm([newX[i] - currentX[i] for i in range(len(currentX))])
        funcShift = abs(newValue - currentValue)

        currentX = newX
        trajectory.append(currentX[::])
        lastStep = step
        iterations += 1

        if pointShift < pointIncEps:
            result = makeResult(currentX, iterations, trajectory, lastStep, "point_shift")
            return result if return_info else result["x"]
        if funcShift < funcIncEps:
            result = makeResult(currentX, iterations, trajectory, lastStep, "func_shift")
            return result if return_info else result["x"]

    result = makeResult(currentX, iterations, trajectory, lastStep, "max_iter")
    return result if return_info else result["x"]
