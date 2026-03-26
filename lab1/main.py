from functions import *
from gradDest import *
from fastGradDest import *
from anal import *

epsilon = 10**(-7)

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


def get_scipy_result(name, startX):
    info = FUNCTIONS_INFO[name]
    if minimize is None:
        return None

    result = minimize(
        info["func"],
        startX,
        method="L-BFGS-B",
        bounds=info["domain"],
    )
    return {"fun": result.fun, "x": result.x}


def run_method_pair(name, startX):
    info = FUNCTIONS_INFO[name]
    gradResult = gradDest(
        info["func"],
        startX,
        epsilon,
        epsilon,
        epsilon,
        100000,
        domain=info["domain"],
        return_info=True,
    )
    fastGradResult = fastGradDest(
        info["func"],
        startX,
        epsilon,
        epsilon,
        epsilon,
        100000,
        domain=info["domain"],
        return_info=True,
    )
    return gradResult, fastGradResult

def print_results(name, analytical, scipy_res, grad, fast_grad):
    print(f"\n=== {name.upper()} ===")
    header = f"{'Method':<16} | {'f(x)':<20} | {'x':<30} | {'iter':<8}"
    print(header)
    print("-" * len(header))

    results = [
        ("Analytical", analytical["value"], analytical["point"], "-"),
        (
            "Scipy",
            scipy_res["fun"] if scipy_res is not None else "N/A",
            scipy_res["x"] if scipy_res is not None else "SciPy is not installed",
            "-",
        ),
        ("Grad Desc", grad["value"], grad["point"], grad["iterations"]),
        ("Fast Grad Desc", fast_grad["value"], fast_grad["point"], fast_grad["iterations"]),
    ]

    for method, val, x, iterations in results:
        print(f"{method:<16} | {str(val):<20} | {str(x):<30} | {iterations}")


for functionName in ("adjiman", "ackley2", "schaffern2"):
    gradResult, fastGradResult = run_method_pair(functionName, [-1, -1])
    scipyResult = get_scipy_result(functionName, [-1, -1])
    print_results(
        functionName,
        analiticalValues[functionName],
        scipyResult,
        {
            "value": FUNCTIONS_INFO[functionName]["func"](gradResult["x"]),
            "point": gradResult["x"],
            "iterations": gradResult["iterations"],
        },
        {
            "value": FUNCTIONS_INFO[functionName]["func"](fastGradResult["x"]),
            "point": fastGradResult["x"],
            "iterations": fastGradResult["iterations"],
        },
    )
