import cv2.cv2 as cv
import numpy as np
from Lib.LVAnalysis.math_functions import *


def detect_axes(img_array, images_out, mid_base, apex):
    """
    Wyznacza położenie i długości osi głównych komory
    :param img_array: obraz maski
    :type img_array: numpy.array
    :param images_out: obrazy wyjściowe
    :type images_out: list
    :param mid_base: punkt położenia środka podstawy
    :type mid_base: tuple/list
    :param apex: punkt położenia koniuszka
    :type apex: tuple/list
    :return: obrazy wyjściowe, długość osi długiej, długość osi krótkiej
    :rtype: list, float, float
    """
    # Linia prostopadła do osi długiej (oś krótka) - punkt przecięcia z osią długą oraz współczynnik nachylenia
    a, mid_x, mid_y = ortogonal(apex, mid_base)
    ort_v = np.vectorize(ortogonal_vector)
    # Linia prostopadła do osi długiej (oś krótka) - wektor
    x = np.arange(1, img_array.shape[1])
    y = ort_v(x, a, mid_x, mid_y).astype(np.int64)
    xy = list(zip(x, y))
    # Ograniczenie długości linii osi krótkiej do wymiarów obrazu
    nxy = [el for el in xy if 0 < el[1] < img_array.shape[0]]
    # Ograniczenie długości linii osi krótkiej do wymiarów komory
    short_axis = [el for el in nxy if img_array[el[1]][el[0]] == 1]
    # Punkty krańców osi
    pt1 = tuple(short_axis[0])
    pt2 = tuple(short_axis[-1])

    for img_out in images_out:
        cv.line(img_out, apex, mid_base, (0, 255, 255), 1, cv.LINE_AA)  # Rysowanie osi długiej
        cv.line(img_out, pt1, pt2, (0, 255, 255), 1, cv.LINE_AA)  # Rysowanie osi krótkiej

    # Obliczanie długości osi długiej oraz krótkiej
    long_axis_length = round(distance_cm(apex, mid_base), 2)
    short_axis_length = round(distance_cm(pt1, pt2), 2)

    return images_out, long_axis_length, short_axis_length
