import csv
import os
import random
from math import sqrt

from anal import analiticalValues
from fastGradDest import fastGradDest
from functions import FUNCTIONS_INFO, rosenbrock
from gradDest import gradDest, projectPoint

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

EPS = 10 ** (-7)
MAX_ITER = 10000
FIXED_STEPS = [0.1, 1.0, 10.0, 100.0]
LINE_SEARCH_EPS_VALUES = [10 ** (-3), 10 ** (-5), 10 ** (-7)]
BASE_STARTS = [(-1.0, -1.0), (2.0, 2.0), (5.0, 5.0), (0.5, 0.5)]


class EvaluationCounter:
    def __init__(self, func):
        self.func = func
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return self.func(x)


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def to_point_str(point):
    return f"({point[0]:.6f}, {point[1]:.6f})"


def format_num(value, digits=6):
    if isinstance(value, str):
        return value
    return f"{value:.{digits}g}"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def write_tex_table(path, caption, label, columns, rows):
    column_spec = "|" + "|".join(["c"] * len(columns)) + "|"
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"\\begin{{longtable}}{{{column_spec}}}\n")
        file.write(f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n")
        file.write("\\hline\n")
        file.write(" & ".join(columns) + " \\\\\n")
        file.write("\\hline\n")
        file.write("\\endfirsthead\n")
        file.write("\\hline\n")
        file.write(" & ".join(columns) + " \\\\\n")
        file.write("\\hline\n")
        file.write("\\endhead\n")
        for row in rows:
            file.write(" & ".join(row) + " \\\\\n")
        file.write("\\hline\n")
        file.write("\\end{longtable}\n")


def run_method(method_name, func, domain, start, **kwargs):
    counted = EvaluationCounter(func)
    used_start = projectPoint(list(start), domain)

    if method_name == "gradDest":
        result = gradDest(
            counted,
            used_start,
            EPS,
            EPS,
            EPS,
            MAX_ITER,
            domain=domain,
            return_info=True,
            **kwargs,
        )
    else:
        result = fastGradDest(
            counted,
            used_start,
            EPS,
            EPS,
            EPS,
            MAX_ITER,
            domain=domain,
            return_info=True,
            **kwargs,
        )

    return {
        "start_requested": tuple(start),
        "start_used": tuple(used_start),
        "point": tuple(result["x"]),
        "value": func(result["x"]),
        "iterations": result["iterations"],
        "func_evals": counted.calls,
        "reason": result["reason"].replace("_", r"\_"),
        "trajectory": result["trajectory"],
        "last_step": result["last_step"],
    }


def run_scipy(func, domain, start):
    if minimize is None:
        return None
    result = minimize(func, start, method="L-BFGS-B", bounds=domain)
    return {
        "point": tuple(result.x),
        "value": result.fun,
        "iterations": result.nit,
        "func_evals": result.nfev,
    }


def get_random_start(domain, seed):
    rnd = random.Random(seed)
    return tuple(rnd.uniform(left, right) for left, right in domain)


def get_hessian(func, point, h=10 ** (-4)):
    n = len(point)
    hessian = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                e = [0.0] * n
                e[i] = h
                plus = [point[k] + e[k] for k in range(n)]
                minus = [point[k] - e[k] for k in range(n)]
                center = func(point)
                hessian[i][i] = (func(plus) - 2 * center + func(minus)) / (h ** 2)
            else:
                e_i = [0.0] * n
                e_j = [0.0] * n
                e_i[i] = h
                e_j[j] = h
                pp = [point[k] + e_i[k] + e_j[k] for k in range(n)]
                pm = [point[k] + e_i[k] - e_j[k] for k in range(n)]
                mp = [point[k] - e_i[k] + e_j[k] for k in range(n)]
                mm = [point[k] - e_i[k] - e_j[k] for k in range(n)]
                hessian[i][j] = (func(pp) - func(pm) - func(mp) + func(mm)) / (4 * h ** 2)

    off_diag = (hessian[0][1] + hessian[1][0]) / 2
    hessian[0][1] = off_diag
    hessian[1][0] = off_diag
    return hessian


def get_eigenvalues_2x2(matrix):
    a = matrix[0][0]
    b = matrix[0][1]
    d = matrix[1][1]
    trace = a + d
    det_part = sqrt((a - d) ** 2 + 4 * b ** 2)
    return ((trace - det_part) / 2, (trace + det_part) / 2)


def get_conditioning_rows():
    rows = []
    for name, info in FUNCTIONS_INFO.items():
        if name == "adjiman":
            point = list(analiticalValues[name]["point"])
            hessian = get_hessian(info["func"], point)
            rows.append([
                info["title"],
                to_point_str(point),
                "полный гессиан неопределён",
                "1.000000 (редуц.)",
                "Минимум лежит на границе $x=2$; для касательного направления по $y$ редуцированный гессиан одномерен, $H_{yy}=3.782735$.",
            ])
            continue
        if name == "ackley2":
            rows.append([
                info["title"],
                to_point_str(analiticalValues[name]["point"]),
                "не определён",
                "не определена",
                "Функция негладкая в точке минимума, так как зависит от $\\sqrt{x^2 + y^2}$.",
            ])
            continue

        point = list(analiticalValues[name]["point"])
        hessian = get_hessian(info["func"], point)
        eig1, eig2 = get_eigenvalues_2x2(hessian)
        eig_abs = sorted([abs(eig1), abs(eig2)])
        condition = eig_abs[1] / eig_abs[0] if eig_abs[0] > 0 else float("inf")
        rows.append([
            info["title"],
            to_point_str(point),
            f"$\\lambda_1={eig1:.6f},\\ \\lambda_2={eig2:.6f}$",
            f"{condition:.6f}",
            f"$H = \\begin{{pmatrix}} {hessian[0][0]:.6f} & {hessian[0][1]:.6f} \\\\ {hessian[1][0]:.6f} & {hessian[1][1]:.6f} \\end{{pmatrix}}$",
        ])
    return rows


def interpolate(p1, v1, p2, v2, level):
    if v1 == v2:
        return p1
    ratio = (level - v1) / (v2 - v1)
    return (
        p1[0] + ratio * (p2[0] - p1[0]),
        p1[1] + ratio * (p2[1] - p1[1]),
    )


def get_contour_segments(xs, ys, values, levels):
    case_to_edges = {
        1: [(3, 0)],
        2: [(0, 1)],
        3: [(3, 1)],
        4: [(1, 2)],
        5: [(3, 2), (0, 1)],
        6: [(0, 2)],
        7: [(3, 2)],
        8: [(2, 3)],
        9: [(0, 2)],
        10: [(0, 3), (1, 2)],
        11: [(1, 2)],
        12: [(1, 3)],
        13: [(0, 1)],
        14: [(3, 0)],
    }
    segments = {level: [] for level in levels}

    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            points = [
                (xs[i], ys[j]),
                (xs[i + 1], ys[j]),
                (xs[i + 1], ys[j + 1]),
                (xs[i], ys[j + 1]),
            ]
            vals = [
                values[j][i],
                values[j][i + 1],
                values[j + 1][i + 1],
                values[j + 1][i],
            ]
            for level in levels:
                mask = 0
                for idx, value in enumerate(vals):
                    if value >= level:
                        mask |= 1 << idx
                if mask == 0 or mask == 15:
                    continue
                for edge_a, edge_b in case_to_edges.get(mask, []):
                    edge_points = [
                        (points[0], vals[0], points[1], vals[1]),
                        (points[1], vals[1], points[2], vals[2]),
                        (points[2], vals[2], points[3], vals[3]),
                        (points[3], vals[3], points[0], vals[0]),
                    ]
                    segment_start = interpolate(*edge_points[edge_a], level)
                    segment_end = interpolate(*edge_points[edge_b], level)
                    segments[level].append((segment_start, segment_end))
    return segments


def create_svg_plot(path, title, func, domain, trajectories):
    width = 880
    height = 620
    margin_left = 70
    margin_right = 30
    margin_top = 60
    margin_bottom = 60

    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    xs = [x_min + (x_max - x_min) * i / 79 for i in range(80)]
    ys = [y_min + (y_max - y_min) * j / 79 for j in range(80)]
    values = [[func([x, y]) for x in xs] for y in ys]

    all_values = [value for row in values for value in row]
    v_min = min(all_values)
    v_max = max(all_values)
    levels = [
        v_min + (v_max - v_min) * level_id / 9
        for level_id in range(1, 9)
    ]
    contours = get_contour_segments(xs, ys, values, levels)

    def map_x(x):
        return margin_left + (x - x_min) * (width - margin_left - margin_right) / (x_max - x_min)

    def map_y(y):
        return height - margin_bottom - (y - y_min) * (height - margin_top - margin_bottom) / (y_max - y_min)

    contour_colors = [
        "#cbd5e1", "#94a3b8", "#64748b", "#475569",
        "#334155", "#1e293b", "#0f172a", "#7f1d1d"
    ]
    trajectory_colors = ["#2563eb", "#dc2626"]

    with open(path, "w", encoding="utf-8") as file:
        file.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        file.write('<rect width="100%" height="100%" fill="#fffdf7"/>\n')
        file.write(f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>\n')
        file.write(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#111827" stroke-width="2"/>\n')
        file.write(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#111827" stroke-width="2"/>\n')

        for tick in range(6):
            tx = x_min + (x_max - x_min) * tick / 5
            px = map_x(tx)
            file.write(f'<line x1="{px}" y1="{height - margin_bottom}" x2="{px}" y2="{height - margin_bottom + 6}" stroke="#111827" stroke-width="1"/>\n')
            file.write(f'<text x="{px}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="13" font-family="Arial">{tx:.2f}</text>\n')

            ty = y_min + (y_max - y_min) * tick / 5
            py = map_y(ty)
            file.write(f'<line x1="{margin_left - 6}" y1="{py}" x2="{margin_left}" y2="{py}" stroke="#111827" stroke-width="1"/>\n')
            file.write(f'<text x="{margin_left - 12}" y="{py + 4}" text-anchor="end" font-size="13" font-family="Arial">{ty:.2f}</text>\n')

        file.write(f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-size="16" font-family="Arial">x</text>\n')
        file.write(f'<text x="20" y="{height / 2}" text-anchor="middle" font-size="16" font-family="Arial" transform="rotate(-90 20 {height / 2})">y</text>\n')

        for idx, level in enumerate(levels):
            color = contour_colors[idx % len(contour_colors)]
            for start, end in contours[level]:
                file.write(
                    f'<line x1="{map_x(start[0]):.2f}" y1="{map_y(start[1]):.2f}" '
                    f'x2="{map_x(end[0]):.2f}" y2="{map_y(end[1]):.2f}" '
                    f'stroke="{color}" stroke-width="1.1" opacity="0.8"/>\n'
                )

        for idx, (label, trajectory) in enumerate(trajectories):
            legend_y = 70 + idx * 24
            if idx == 0:
                file.write(f'<rect x="{width - 270}" y="48" width="210" height="56" fill="#ffffff" stroke="#d1d5db"/>\n')
            color = trajectory_colors[idx % len(trajectory_colors)]
            points = " ".join(f"{map_x(point[0]):.2f},{map_y(point[1]):.2f}" for point in trajectory)
            file.write(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>\n')
            for point in trajectory:
                file.write(f'<circle cx="{map_x(point[0]):.2f}" cy="{map_y(point[1]):.2f}" r="3.5" fill="{color}"/>\n')
            file.write(f'<line x1="{width - 250}" y1="{legend_y}" x2="{width - 220}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>\n')
            file.write(f'<text x="{width - 210}" y="{legend_y + 4}" font-size="14" font-family="Arial">{label}</text>\n')
        file.write('</svg>\n')


def create_tikz_plot(path, func, domain, trajectories):
    x_min, x_max = domain[0]
    y_min, y_max = domain[1]
    xs = [x_min + (x_max - x_min) * i / 59 for i in range(60)]
    ys = [y_min + (y_max - y_min) * j / 59 for j in range(60)]
    values = [[func([x, y]) for x in xs] for y in ys]

    all_values = [value for row in values for value in row]
    v_min = min(all_values)
    v_max = max(all_values)
    levels = [
        v_min + (v_max - v_min) * level_id / 7
        for level_id in range(1, 7)
    ]
    contours = get_contour_segments(xs, ys, values, levels)

    plot_width = 12.0
    plot_height = 8.0

    def map_x(x):
        return (x - x_min) * plot_width / (x_max - x_min)

    def map_y(y):
        return (y - y_min) * plot_height / (y_max - y_min)

    contour_colors = ["gray!35", "gray!45", "gray!55", "gray!65", "gray!75", "gray!85"]
    trajectory_styles = [("blue!80!black", "gradDest"), ("red!80!black", "fastGradDest")]

    with open(path, "w", encoding="utf-8") as file:
        file.write("\\begin{tikzpicture}[scale=0.9]\n")
        file.write(f"\\draw[fill=orange!3, draw=black] (0,0) rectangle ({plot_width:.4f},{plot_height:.4f});\n")
        for idx, level in enumerate(levels):
            color = contour_colors[idx % len(contour_colors)]
            for start, end in contours[level]:
                file.write(
                    f"\\draw[{color}, line width=0.15pt] "
                    f"({map_x(start[0]):.4f},{map_y(start[1]):.4f}) -- "
                    f"({map_x(end[0]):.4f},{map_y(end[1]):.4f});\n"
                )

        for index, (label, trajectory) in enumerate(trajectories):
            color, _ = trajectory_styles[index % len(trajectory_styles)]
            points = " -- ".join(f"({map_x(point[0]):.4f},{map_y(point[1]):.4f})" for point in trajectory)
            file.write(f"\\draw[{color}, line width=1.0pt] {points};\n")
            for point in trajectory:
                file.write(f"\\fill[{color}] ({map_x(point[0]):.4f},{map_y(point[1]):.4f}) circle (1.2pt);\n")

        file.write(f"\\draw[->, thick] (0,0) -- ({plot_width + 0.6:.4f},0) node[right] {{$x$}};\n")
        file.write(f"\\draw[->, thick] (0,0) -- (0,{plot_height + 0.6:.4f}) node[above] {{$y$}};\n")

        for tick in range(6):
            tx = x_min + (x_max - x_min) * tick / 5
            px = map_x(tx)
            file.write(f"\\draw ({px:.4f},0) -- ({px:.4f},-0.12) node[below] {{\\small {tx:.2f}}};\n")
            ty = y_min + (y_max - y_min) * tick / 5
            py = map_y(ty)
            file.write(f"\\draw (0,{py:.4f}) -- (-0.12,{py:.4f}) node[left] {{\\small {ty:.2f}}};\n")

        file.write("\\draw[fill=white, draw=black] (7.6,6.7) rectangle (11.8,7.9);\n")
        file.write("\\draw[blue!80!black, line width=1.0pt] (7.9,7.45) -- (8.9,7.45);\n")
        file.write("\\node[anchor=west] at (9.1,7.45) {\\small gradDest};\n")
        file.write("\\draw[red!80!black, line width=1.0pt] (7.9,7.05) -- (8.9,7.05);\n")
        file.write("\\node[anchor=west] at (9.1,7.05) {\\small fastGradDest};\n")
        file.write("\\end{tikzpicture}\n")


def build_start_rows():
    csv_rows = []
    tex_rows = []

    for index, (name, info) in enumerate(FUNCTIONS_INFO.items()):
        starts = BASE_STARTS + [get_random_start(info["domain"], 100 + index)]
        for start in starts:
            for method_name, method_kwargs in (
                ("gradDest", {"t": 1.0}),
                ("fastGradDest", {"lineSearchEps": 10 ** (-5), "initialStep": 1.0}),
            ):
                result = run_method(method_name, info["func"], info["domain"], start, **method_kwargs)
                csv_rows.append([
                    info["title"],
                    method_name,
                    to_point_str(start),
                    to_point_str(result["start_used"]),
                    to_point_str(result["point"]),
                    f"{result['value']:.10f}",
                    result["iterations"],
                    result["func_evals"],
                    result["reason"],
                ])
                tex_rows.append([
                    info["title"],
                    method_name,
                    to_point_str(start),
                    to_point_str(result["point"]),
                    f"{result['value']:.6f}",
                    str(result["iterations"]),
                    str(result["func_evals"]),
                ])

    write_csv(
        os.path.join(TABLES_DIR, "starts.csv"),
        ["function", "method", "start_requested", "start_used", "point", "value", "iterations", "func_evals", "reason"],
        csv_rows,
    )
    write_tex_table(
        os.path.join(TABLES_DIR, "starts.tex"),
        "Влияние начальной точки на сходимость методов.",
        "tab:starts",
        ["Функция", "Метод", "Старт", "Найденная точка", "$f(x)$", "Итерации", "Вызовы $f$"],
        tex_rows,
    )


def build_hyperparameter_rows():
    grad_csv_rows = []
    grad_tex_rows = []
    fast_csv_rows = []
    fast_tex_rows = []

    for name, info in FUNCTIONS_INFO.items():
        start = (-1.0, -1.0)
        for step in FIXED_STEPS:
            result = run_method("gradDest", info["func"], info["domain"], start, t=step)
            grad_csv_rows.append([
                info["title"],
                f"{step:g}",
                to_point_str(result["point"]),
                f"{result['value']:.10f}",
                result["iterations"],
                result["func_evals"],
                result["reason"],
            ])
            grad_tex_rows.append([
                info["title"],
                f"{step:g}",
                f"{result['value']:.6f}",
                str(result["iterations"]),
                str(result["func_evals"]),
                result["reason"],
            ])

        for line_eps in LINE_SEARCH_EPS_VALUES:
            result = run_method(
                "fastGradDest",
                info["func"],
                info["domain"],
                start,
                lineSearchEps=line_eps,
                initialStep=1.0,
            )
            fast_csv_rows.append([
                info["title"],
                f"{line_eps:.0e}",
                to_point_str(result["point"]),
                f"{result['value']:.10f}",
                result["iterations"],
                result["func_evals"],
                result["reason"],
            ])
            fast_tex_rows.append([
                info["title"],
                f"{line_eps:.0e}",
                f"{result['value']:.6f}",
                str(result["iterations"]),
                str(result["func_evals"]),
                result["reason"],
            ])

    write_csv(
        os.path.join(TABLES_DIR, "grad_tuning.csv"),
        ["function", "t", "point", "value", "iterations", "func_evals", "reason"],
        grad_csv_rows,
    )
    write_tex_table(
        os.path.join(TABLES_DIR, "grad_tuning.tex"),
        "Влияние начального шага $t$ на метод с дроблением шага.",
        "tab:grad_tuning",
        ["Функция", "$t$", "$f(x)$", "Итерации", "Вызовы $f$", "Причина остановки"],
        grad_tex_rows,
    )

    write_csv(
        os.path.join(TABLES_DIR, "fast_tuning.csv"),
        ["function", "line_search_eps", "point", "value", "iterations", "func_evals", "reason"],
        fast_csv_rows,
    )
    write_tex_table(
        os.path.join(TABLES_DIR, "fast_tuning.tex"),
        "Влияние точности одномерного поиска на метод с линейным поиском.",
        "tab:fast_tuning",
        ["Функция", "$\\varepsilon_{ls}$", "$f(x)$", "Итерации", "Вызовы $f$", "Причина остановки"],
        fast_tex_rows,
    )


def build_analytical_rows():
    rows = []
    for name, info in FUNCTIONS_INFO.items():
        rows.append([
            info["title"],
            f"${info['formula']}$",
            f"$[{info['domain'][0][0]:g}, {info['domain'][0][1]:g}] \\times [{info['domain'][1][0]:g}, {info['domain'][1][1]:g}]$",
            to_point_str(analiticalValues[name]["point"]),
            f"{analiticalValues[name]['value']:.10f}",
        ])

    write_tex_table(
        os.path.join(TABLES_DIR, "analytical.tex"),
        "Выбранные тестовые функции и их эталонные минимумы.",
        "tab:analytical",
        ["Функция", "Формула", "Область", "Точка минимума", "Минимум"],
        rows,
    )


def build_conditioning_table():
    rows = get_conditioning_rows()
    write_tex_table(
        os.path.join(TABLES_DIR, "conditioning.tex"),
        "Число обусловленности в окрестности минимума.",
        "tab:conditioning",
        ["Функция", "Точка", "Собственные числа", "Число обусловленности", "Комментарий"],
        rows,
    )


def build_scipy_table():
    rows = []
    for name, info in FUNCTIONS_INFO.items():
        scipy_result = run_scipy(info["func"], info["domain"], [-1.0, -1.0])
        if scipy_result is None:
            rows.append([info["title"], "SciPy недоступен в текущем окружении", "-", "-", "-"])
        else:
            rows.append([
                info["title"],
                to_point_str(scipy_result["point"]),
                f"{scipy_result['value']:.10f}",
                str(scipy_result["iterations"]),
                str(scipy_result["func_evals"]),
            ])

    write_tex_table(
        os.path.join(TABLES_DIR, "scipy.tex"),
        "Сравнение с библиотечной реализацией scipy.optimize.minimize.",
        "tab:scipy",
        ["Функция", "Точка", "$f(x)$", "Итерации", "Вызовы $f$"],
        rows,
    )


def build_plots():
    for name, info in FUNCTIONS_INFO.items():
        start = (-1.0, -1.0)
        grad_result = run_method("gradDest", info["func"], info["domain"], start, t=1.0)
        fast_result = run_method(
            "fastGradDest",
            info["func"],
            info["domain"],
            start,
            lineSearchEps=10 ** (-5),
            initialStep=1.0,
        )
        create_svg_plot(
            os.path.join(PLOTS_DIR, f"{name}.svg"),
            f"{info['title']}: линии уровня и траектории",
            info["func"],
            info["domain"],
            [
                ("gradDest", grad_result["trajectory"]),
                ("fastGradDest", fast_result["trajectory"]),
            ],
        )
        create_tikz_plot(
            os.path.join(PLOTS_DIR, f"{name}.tex"),
            info["func"],
            info["domain"],
            [
                ("gradDest", grad_result["trajectory"]),
                ("fastGradDest", fast_result["trajectory"]),
            ],
        )

def build_rosenbrock_table():
    info = FUNCTIONS_INFO["rosenbrock"]
    b_values = [1, 10, 100, 1000]
    start = (-2.0, -2.0)

    rows = []
    for b in b_values:
        func = lambda x: rosenbrock(x, b=b)
        for method_name, method_kwargs in (
            ("gradDest", {"t": 1.0}),
            ("fastGradDest", {"lineSearchEps": 1e-5, "initialStep": 1.0}),
        ):
            result = run_method(method_name, func, info["domain"], start, **method_kwargs)
            rows.append([
                f"${b}$",
                method_name,
                to_point_str(result["point"]),
                f"{result['value']:.6f}",
                str(result["iterations"]),
                str(result["func_evals"]),
                result["reason"],
            ])

    write_tex_table(
        os.path.join(TABLES_DIR, "rosenbrock_tuning.tex"),
        "Влияние параметра овражности $b$ на сходимость.",
        "tab:rosenbrock",
        ["Параметр $b$", "Метод", "Найденная точка", "$f(x)$", "Итерации", "Вызовы $f$", "Причина"],
        rows,
    )

def build_rosenbrock_plot():
    info = FUNCTIONS_INFO["rosenbrock"]
    b = 100
    func = lambda x: rosenbrock(x, b=b)
    start = (-2.0, -2.0)

    grad_result = run_method("gradDest", func, info["domain"], start, t=1.0)
    fast_result = run_method("fastGradDest", func, info["domain"], start,
                             lineSearchEps=1e-5, initialStep=1.0)

    create_tikz_plot(
        os.path.join(PLOTS_DIR, "rosenbrock.tex"),
        func,
        info["domain"],
        [
            ("gradDest", grad_result["trajectory"]),
            ("fastGradDest", fast_result["trajectory"]),
        ],
    )
    create_svg_plot(
        os.path.join(PLOTS_DIR, "rosenbrock.svg"),
        f"{info['title']} (b={b}): линии уровня и траектории",
        func,
        info["domain"],
        [
            ("gradDest", grad_result["trajectory"]),
            ("fastGradDest", fast_result["trajectory"]),
        ],
    )


def generate_all():
    ensure_dirs()
    build_analytical_rows()
    build_start_rows()
    build_hyperparameter_rows()
    build_conditioning_table()
    build_scipy_table()
    build_plots()
    build_rosenbrock_table()
    build_rosenbrock_plot()


if __name__ == "__main__":
    generate_all()
