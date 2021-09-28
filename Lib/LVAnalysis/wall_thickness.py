from Lib.LVAnalysis.wall_lines_length import wall_lines_length
from Lib.LVAnalysis.math_functions import *


def count_wall_thickness(contour1, contour2, num_parts=8):
    """
    Funkcja oblicza grubość ściany metodą podziału na równe części obu krawędzi ściany
    :param contour1: pierwsza krawędź (linia) ściany
    :type contour1: list
    :param contour2: druga krawędź (linia) ściany
    :type contour2: list
    :param num_parts: liczba części
    :type num_parts: int
    :return: avg_thiskness: średnia grubość ściany
    :rtype: float
    """
    sum_thick = 0
    pts1 = []
    pts2 = []
    len1 = wall_lines_length(contour1, 0, len(contour1) - 1)
    len2 = wall_lines_length(contour2, 0, len(contour2) - 1)

    # Jeżeli długości będą dostatecznie małe, liczba części zostaje zmniejszona
    if len1 < 0.5 or len2 < 0.5:
        num_parts = 4
        if len1 < 0.2 or len2 < 0.2:
            num_parts = 2
    # Obliczanie kroku
    step1 = round(len1 / num_parts, 2)
    step2 = round(len2 / num_parts, 2)
    # Zmienna k oznacza wielokrotność kroku
    k = 1
    for j in range(0, len(contour1) - 1):
        length = wall_lines_length(contour1, 0, j, 5)
        # Jeżeli długość przekroczyła daną k-wielokrotność kroku a ilość znalezionych punktów jest równa k-1
        if length > k * step1 and len(pts1) == k - 1:
            pts1.append(contour1[j - 1])
            k = k + 1
            if k == num_parts:
                break
    # Analogiczne działanie dla konturu 2:
    k = 1
    for j in range(0, len(contour2) - 1):
        length = wall_lines_length(contour2, 0, j, 5)
        if length > k * step2 and len(pts2) == k - 1:
            pts2.append(contour2[j - 1])
            k = k + 1
            if k == num_parts:
                break
    if len(pts1) == 0 or len(pts2) == 0:
        return
    else:
        if len(pts1) <= len(pts2):
            counted = len(pts1)
        else:
            counted = len(pts2)
        for j in range(0, counted):
            sum_thick = sum_thick + distance_cm(pts1[j], pts2[j])

        avg_thickness = round(sum_thick / len(pts1), 2)

    return avg_thickness