from Lib.LVAnalysis.wall_line_intersection import wall_line_intersection
from Lib.LVAnalysis.math_functions import *
import numpy as np


def count_wall_thickness_ortogonal(image, contour1, contour2):
    """
       Funkcja oblicza grubość ściany metodą podziału na równe części obu krawędzi ściany
       :param image: obraz
       :type image: array
       :param contour1: wewnętrna krawędź (linia) ściany
       :type contour1: list
       :param contour2: zewnętrzna krawędź (linia) ściany
       :type contour2: list
       :return: avg_thiskness: średnia grubość ściany
       :rtype: float
       """
    sum_thick = 0
    counter = 0
    line_v = np.vectorize(ortogonal_vector)
    for i in range(0, len(contour1)-4, 5):
        if i+5 > len(contour1)-1:
            a, x0, y0 = ortogonal(contour1[i], contour1[-1])
        else:
            a, x0, y0 = ortogonal(contour1[i], contour1[i + 5])
        if a is None:
            y = np.arange(y0 - 140, y0 + 40)
            x = line_v(y, a, x0, y0).astype(np.int64)
        if a is not None:
            x = np.arange(x0 - 140, x0 + 140)
            y = line_v(x, a, x0, y0).astype(np.int64)
        nx = []
        ny = []
        for i in range(len(x)):
            if image.shape[1] > x[i] > 0 and 0 < y[i] < image.shape[0]:
                nx.append(x[i])
                ny.append(y[i])
        pt = wall_line_intersection(contour2, (nx[0], ny[0]), (nx[-1], ny[-1]), 3)
        if pt:
            counter = counter + 1
            sum_thick = sum_thick+distance_cm(pt["pt"], (x0, y0))

    if counter != 0:
        avg_thickness = round(sum_thick/counter, 2)
    else:
        avg_thickness = None

    return avg_thickness

