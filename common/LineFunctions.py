import numpy as np


def slope_of_line(line: np.ndarray):
    delta = 1e-6
    if line.ndim == 2:
        line = line[0]
    return (line[3] - line[1]) / (line[2] - line[0] + delta)


def length_of_line(line: np.ndarray):
    if line.ndim == 2:
        line = line[0]
    return ((line[3] - line[1])**2 + (line[2] - line[0])**2)**(1/2)


def value_of_line(point: tuple, line: np.ndarray):
    delta = 1e-6
    if line.ndim == 2:
        line = line[0]
    if line[3] - line[1] < delta:
        return point[1] - line[3]
    elif line[2] - line[0] < delta:
        return point[0] - line[2]
    else:
        return (point[1] - line[1]) / (line[3] - line[1]) - (point[0] - line[0]) / (line[2] - line[0])


def distance_point_line(point: tuple, line: np.ndarray):
    if line.ndim == 2:
        line = line[0]

    # vector 1
    x1, y1 = point[0] - line[0], point[1] - line[1]
    # vector 2
    x2, y2 = (line[2] - line[0], line[3] - line[1])
    assert (x2 ** 2 + y2 ** 2) > 0
    return abs(x1 * y2 - x2 * y1) / (x2 ** 2 + y2 ** 2) ** (1 / 2)


def line_diff_old(line: np.ndarray, line_base: np.ndarray):
    if line.ndim == 2:
        line = line[0]
    return value_of_line((line[0], line[1]), line_base)**2 + value_of_line((line[2], line[3]), line_base)**2


def line_diff(line_1: np.ndarray, line_2: np.ndarray):
    deg = 0.25
    if line_1.ndim == 2:
        line_1 = line_1[0]
    if line_2.ndim == 2:
        line_2 = line_2[0]

    if abs(slope_of_line(line_2) - slope_of_line(line_1)) < deg:
        return True
        # if abs(value_of_line((line_2[0], line_2[1]), line_1)) < deg:
        #     return True
        # else:
        #     return False
    else:
        return False


def intersection_point(line_1: np.ndarray, line_2: np.ndarray):
    pass


def perpendicular_feet(point: tuple, line: np.ndarray):
    if line.ndim == 2:
        line = line[0]
    x_1, y_1, x_2, y_2 = line
    x_0, y_0 = point

    m = 1 / (x_2 - x_1)
    n = 1 / (y_2 - y_1)

    a = np.array([[m, -n], [n * (x_2 - x_1), 1]])
    b = np.array([m * x_1 - n * y_1, y_0 + n * x_0 * (x_2 - x_1)])

    xy = np.linalg.solve(a, b)
    return xy
