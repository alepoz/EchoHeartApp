import cv2 as cv
import numpy as np
from Lib.LVAnalysis.math_functions import *


def simpson_disks_ventricular(img_array, data, num_of_disks=20):
    """
    Dzieli komorę na dyski, oblicza średnice dysków oraz ich wysokości
    :param img_array: obraz wejściowy (maska)
    :type img_array: array
    :param data: dane parametrów - położenie koniuszka i punktów podstaw
    :type data: dict
    :param num_of_disks: liczba dysków
    :type num_of_disks: int
    :return: średnice dysków, wysokości dysków
    :rtype: list, list
    """
    # Importowanie danych
    det1 = tuple(data['base_left'])
    mid_base = tuple(data['base_mid'])
    det2 = tuple(data['base_right'])
    apex = tuple(data['apex'])

    # Wektor(linia) dzieląca obraz wdłuż osi głównej na część lewą i prawą
    y = np.arange(0, img_array.shape[0])
    line_v = np.vectorize(line_vector)
    x = line_v(y, apex[1], mid_base[1], apex[0], mid_base[0]).astype(np.uint)
    xy = list(zip(x, y))

    # Usuwanie części linii wykraczających poza wymiary obrazu
    nxy = [el for el in xy if 0 <= el[0] <= img_array.shape[1]]

    # Współczynnik nachylenia podstawy i równoległych do niej dysków
    a = (det2[1] - det1[1]) / (det2[0] - det1[0])

    line_base = np.vectorize(ortogonal_vector)
    for j in range(0, len(nxy) - 1):
        # Linie prostopadłe do linii osi długiej komory
        bx = np.arange(1, img_array.shape[1])
        by = line_base(bx, a, nxy[j][0], nxy[j][1]).astype(np.uint)
        bxy = list(zip(bx, by))
        # ograniczenie linii prostopadłych do wymiarów obrazu
        nbxy = [el for el in bxy if 0 < el[1] < img_array.shape[0]]
        # Sprawdzenie czy linia prostopadła przechodzi przez obszar komory
        vec_disk = [el for el in nbxy if img_array[el[1]][el[0]] == 1]
        if vec_disk:  # Jeżeli linia pzechodzi przez komorę to punkt osi, przez który przechodzi jest punktem startowym
            start_pt = (nxy[j][0], nxy[j][1])
            break

    axis_pts = []
    step = 1 / (2 * num_of_disks)
    for t in np.arange(0, 1.0001, step):  # podział na num_of_disks dysków
        ax = int(start_pt[0] * (1 - t) + mid_base[0] * t)
        ay = int(start_pt[1] * (1 - t) + mid_base[1] * t)
        axis_pts.append((ax, ay))

    diameters = []
    # podstawa dysku - punkt A
    baseA = []
    # podstawa dysku - punkt B
    baseB = []
    # linie przebiegające przez krawędzie i linie środkowe dysków
    for j in range(0, len(axis_pts)):
        bx = np.arange(1, img_array.shape[1])
        by = line_base(bx, a, axis_pts[j][0], axis_pts[j][1]).astype(np.uint)
        nbx = []
        nby = []
        # ograniczenie lini dysków do wymiarów obrazu
        for i in range(len(bx)):
            if img_array.shape[1] > bx[i] > 0 and 0 < by[i] < img_array.shape[0]:
                nbx.append(bx[i])
                nby.append(by[i])
        nbxy = list(zip(nbx, nby))

        # ograniczenie linii dysków do wymiarów komory
        base_disk = [el for el in nbxy if img_array[el[1]][el[0]] == 1]
        px_1 = None
        px_2 = None
        # Jeżeli wykryto ścianę
        if base_disk:
            # podstawa dysku
            px_1 = (int(base_disk[0][0]), int(base_disk[0][1]))
            px_2 = (int(base_disk[-1][0]), int(base_disk[-1][1]))
            if j % 2 == 0:
                # podstawa dysku
                baseA.append(px_1)
                baseB.append(px_2)
            else:
                # średnica dysku
                diameters.append(distance_cm(px_1, px_2))  # długość średnicy w cm

    # Wysokość dysku
    heights = []
    for (p1, p2, E) in zip(baseA[1::], baseB[1::], axis_pts[::2]):
        # E - punkt na osi długiej, prez który przechodzi granica dysku
        x0 = E[0]
        y0 = E[1]
        # Wyznaczenie współczynników prostej: Ax + By + C = 0
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0] * p2[1] - p2[0] * p1[1]
        # xP,yP - współrzędne punktu, na który spada wysokość dysku
        xP = (B * (B * x0 - A * y0) - A * C) / (A ** 2 + B ** 2)
        yP = (A * (-B * x0 + A * y0) - B * C) / (A ** 2 + B ** 2)
        h = distance_cm((x0, y0), (int(xP), int(yP)))
        heights.append(h)

    return diameters, heights


def simpson_disks_wall(img_array, data):
    """
        Dzieli komorę wraz ze ścianą na dyski, oblicza średnice dysków oraz ich wysokości
        :param img_array: obraz wejściowy (maska)
        :type img_array: array
        :param data: dane parametrów - położenie koniuszka i punktów podstaw
        :type data: dict
        :param num_of_disks: liczba dysków
        :type num_of_disks: int
        :return: średnice dysków, wysokości dysków
        :rtype: list, list
        """
    # Import danych
    det1 = tuple(data['base_left'])
    mid_base = tuple(data['base_mid'])
    det2 = tuple(data['base_right'])
    apex = tuple(data['apex'])

    # Maska ściany
    wall_mask = cv.inRange(img_array, 2, 2)
    _, thresh_wall = cv.threshold(wall_mask, 2, 1, cv.THRESH_BINARY)
    contours_wall, _ = cv.findContours(thresh_wall, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Kontur ściany

    # Zamknięcie maski ściany w wielokącie
    hull_mask = np.zeros((img_array.shape[0], img_array.shape[1]), np.uint8)
    hull_contour = []
    for i in range(len(contours_wall)):
        hull_contour.append(cv.convexHull(contours_wall[i], False))
    cv.drawContours(hull_mask, hull_contour, -1, 255, thickness=-1)
    hull_mask_c = cv.cvtColor(hull_mask, cv.COLOR_GRAY2BGR)

    # Wektor(linia) dzieląca obraz wdłuż osi głównej na część lewą i prawą
    y = np.arange(0, img_array.shape[0])
    line_v = np.vectorize(line_vector)
    x = line_v(y, apex[1], mid_base[1], apex[0], mid_base[0]).astype(np.uint)
    xy = list(zip(x, y))

    # Usuwanie części linii wykraczających poza wymiary obrazu
    axis = [el for el in xy if 0 <= el[0] <= img_array.shape[1]]

    # Współczynnik nachylenia podstawy i równoległych do niej dysków
    a = (det2[1] - det1[1]) / (det2[0] - det1[0])

    line_base = np.vectorize(ortogonal_vector)
    for j in range(0, len(axis) - 1):
        px_1 = None
        px_2 = None
        # Linie prostopadłe do linii osi długiej komory
        bx = np.arange(1, img_array.shape[1])
        by = line_base(bx, a, axis[j][0], axis[j][1]).astype(np.uint)
        bxy = list(zip(bx, by))
        # ograniczenie linii prostopadłych do wymiarów obrazu
        nbxy = [el for el in bxy if 0 < el[1] < img_array.shape[0]]
        # Sprawdzenie czy linia prostopadła przechodzi przez obszar komory
        vec_disk = [el for el in nbxy if hull_mask[el[1]][el[0]] == 255]
        if vec_disk:  # Jeżeli linia pzechodzi przez komorę to punkt osi, przez który przechodzi jest punktem startowym
            start_pt = (axis[j][0], axis[j][1])
            break

    axis_pts = []
    num_of_disks = 40
    step = 1 / (2 * num_of_disks)
    for t in np.arange(0, 1.0001, step):  # podział na 40 dysków
        ax = int(start_pt[0] * (1 - t) + mid_base[0] * t)
        ay = int(start_pt[1] * (1 - t) + (mid_base[1]-1) * t)
        axis_pts.append((ax, ay))

    diameters = []
    # podstawa dysku - punkt A
    baseA = []
    # podstawa dysku - punkt B
    baseB = []
    # linie przebiegające przez krawędzie i linie środkowe dysków
    for j in range(0, len(axis_pts)):
        px_1 = None
        px_2 = None
        bx = np.arange(1, img_array.shape[1])
        by = line_base(bx, a, axis_pts[j][0], axis_pts[j][1]).astype(np.uint)
        nbx = []
        nby = []
        # ograniczenie lini dysków do wymiarów obrazu
        for i in range(len(bx)):
            if img_array.shape[1] > bx[i] > 0 and 0 < by[i] < img_array.shape[0]:
                nbx.append(bx[i])
                nby.append(by[i])

        nbxy = list(zip(nbx, nby))
        # ograniczenie linii dysków do wymiarów komory
        base_disk = [el for el in nbxy if hull_mask[el[1]][el[0]] == 255]
        # Jeżeli wykryto ścianę
        if base_disk:
            # granica dysku
            px_1 = (int(base_disk[0][0]), int(base_disk[0][1]))
            px_2 = (int(base_disk[-1][0]), int(base_disk[-1][1]))
            if j % 2 == 0:
                # podstawa dysku
                baseA.append(px_1)
                baseB.append(px_2)
            else:
                # średnica dysku
                diameters.append(distance_cm(px_1, px_2))  # długość średnicy w cm

    # Wysokość dysku
    heights = []
    for (p1, p2, E) in zip(baseA[1::], baseB[1::], axis_pts[::2]):
        # E - punkt na osi długiej, prez który przechodzi granica dysku
        x0 = E[0]
        y0 = E[1]
        A = p1[1] - p2[1]
        B = p2[0] - p1[0]
        C = p1[0] * p2[1] - p2[0] * p1[1]
        # xP,yP - współrzędne punktu podstawy dysku, na który spada wysokość dysku
        xP = (B * (B * x0 - A * y0) - A * C) / (A ** 2 + B ** 2)
        yP = (A * (-B * x0 + A * y0) - B * C) / (A ** 2 + B ** 2)
        h = distance_cm((x0, y0), (int(xP), int(yP)))
        heights.append(h)
        cv.line(hull_mask_c, E, (int(xP), int(yP)), (0, 255, 0), 1, cv.LINE_AA)

    return diameters, heights
