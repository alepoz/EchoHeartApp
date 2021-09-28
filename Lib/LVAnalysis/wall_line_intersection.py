from Lib.LVAnalysis.math_functions import *


def wall_line_intersection(contour, pt1, pt2, step=5):
    """
    Funkcja wyznacza punkt przecięcia krzywej z linią prostą
    :param contour: krzywa
    :type contour: list
    :param pt1: punkt początkowy prostej
    :type pt1: tuple/list
    :param pt2: punkt końcowy prostej
    :type pt2: tuple/list
    :param step: krok
    :type step: int
    :return: sec_point: słownik przechowujący współrzędne punktu pod kluczem 'pt' oraz indeks w liście contour pod
    kluczem 'idx'
    :rtype: dict
    """
    sect_point = {}
    for i in range(0, len(contour) - step, step):
        if i + step > len(contour) - step:
            sect_pt = line_intersect([contour[i], contour[-1]], [pt1, pt2])
        else:
            sect_pt = line_intersect([contour[i], contour[i + step]], [pt1, pt2])
        if sect_pt is not None:
            min_diff = None
            for j, pt in enumerate(contour[i:i + step]):
                diff = distance(sect_pt, pt)
                if min_diff is None:
                    min_diff = diff
                    sect_point["pt"] = tuple(pt)
                    sect_point["idx"] = i + j
                elif diff < min_diff:
                    sect_point["pt"] = tuple(pt)
                    sect_point["idx"] = i + j
    if sect_point:
        return sect_point
    else:
        return None
