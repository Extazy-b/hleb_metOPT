from math import *

def adjiman(x):
    x1, x2 = x
    return cos(x1) * sin(x2) - x1 / (x2**2 + 1)

def ackley2(x):
    x1, x2 = x
    return -200 * exp(-0.02 * sqrt(x1**2 + x2**2))

def schaffern2(x):
    x1, x2 = x
    num = sin(x1**2 - x2**2)**2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + num / den

def rosenbrock(x, b=100):
    return (1 - x[0])**2 + b * (x[1] - x[0]**2)**2


FUNCTIONS_INFO = {
    "adjiman": {
        "func": adjiman,
        "title": "Adjiman",
        "domain": [(-1.0, 2.0), (-1.0, 1.0)],
        "formula": r"f(x, y) = \cos(x)\sin(y) - \frac{x}{1 + y^2}",
    },
    "ackley2": {
        "func": ackley2,
        "title": "Ackley 2",
        "domain": [(-32.0, 32.0), (-32.0, 32.0)],
        "formula": r"f(x, y) = -200 \exp \left(-0.02 \sqrt{x^2 + y^2}\right)",
    },
    "schaffern2": {
        "func": schaffern2,
        "title": "Schaffer N.2",
        "domain": [(-100.0, 100.0), (-100.0, 100.0)],
        "formula": r"f(x, y) = 0.5 + \frac{\sin^2(x^2 - y^2) - 0.5}{(1 + 0.001(x^2 + y^2))^2}",
    },    
    "rosenbrock": {
        "func": rosenbrock,
        "title": "Rosenbrock",
        "domain": [(-5.0, 5.0), (-5.0, 5.0)],
        "formula": r"f(x, y) = (1 - x)^2 + b\,(y - x^2)^2",
    },
}
