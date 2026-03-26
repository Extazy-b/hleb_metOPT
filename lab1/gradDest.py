from myMath import *

def projectPoint(x, domain):
    if domain is None:
        return x[::]
    return [
        min(max(x[i], domain[i][0]), domain[i][1])
        for i in range(len(x))
    ]


def makeResult(x, iterations, path, lastStep, reason):
    return {
        "x": x[::],
        "iterations": iterations,
        "trajectory": [point[::] for point in path],
        "last_step": lastStep,
        "reason": reason,
    }


def gradDest(
    fun,
    startX,
    gradNormEps,
    pointIncEps,
    funcIncEps,
    MaxIter,
    t=1,
    minStep=10 ** (-12),
    domain=None,
    return_info=False,
):
    currentX = projectPoint(startX, domain)
    trajectory = [currentX[::]]
    iterations = 0
    lastStep = 0

    while iterations < MaxIter:
        gr = getGrad(fun, currentX)
        if norm(gr) < gradNormEps:
            result = makeResult(currentX, iterations, trajectory, lastStep, "grad_norm")
            return result if return_info else result["x"]

        currentValue = fun(currentX)
        step = t
        candidateX = currentX[::]
        candidateValue = currentValue

        while step >= minStep:
            trialX = [currentX[i] - step * gr[i] for i in range(len(currentX))]
            trialX = projectPoint(trialX, domain)
            trialValue = fun(trialX)
            if trialValue <= currentValue:
                candidateX = trialX
                candidateValue = trialValue
                break
            step /= 2

        if step < minStep:
            result = makeResult(currentX, iterations, trajectory, lastStep, "step_underflow")
            return result if return_info else result["x"]

        pointShift = norm([candidateX[i] - currentX[i] for i in range(len(currentX))])
        funcShift = abs(candidateValue - currentValue)

        currentX = candidateX
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
