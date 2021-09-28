from Lib.LVAnalysis.math_functions import *

def wall_lines_length(contour, idx1, idx2, step=10):
    """
    Funkcja obliczająca długość konturu(linii ścian) lub jego fragmentu
    :param contour: kontur(linia) ściany
    :type contour: list
    :param idx1: indeks punktu początkowego w konturze, dla którego liczona jest długość
    :type idx1: int
    :param idx2: indeks punktu końcowego w konturze,  dla którego liczona jest długość
    :type idx2: int
    :param step: krok (między punktami wyznaczającymi odcinki)
    :type step: int
    :return: line_length: długość linii
    :rtype: float
    """
    line_length = 0
    for j in range(idx1, idx2, step):
        if j + step >= idx2:
            line_length = line_length + distance_cm(contour[j], contour[idx2])
            break
        else:
            line_length = line_length + distance_cm(contour[j], contour[j + step])

    return line_length
