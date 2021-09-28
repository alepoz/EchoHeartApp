import cv2.cv2 as cv
import numpy as np
from shapely.geometry import Point, Polygon
from scipy import signal
from Lib.LVAnalysis.wall_lines_length import wall_lines_length
from Lib.LVAnalysis.math_functions import *


def detect_wall_lines(img_array, images_out):
    """
    :param img_array: obraz
    :type img_array: numpy.array
    :param images_out: obrazy wyjściowe
    :type images_out: list
    :returns: obrazy wyjściowe, punkty linii epicardialnej, punkty linii środkowej, punkty linii endocardialnej, oraz
    kolejno długości tych linii
    :rtype: list, list, list, list, float, float, float
    """
    # Maska ściany
    wall_mask = cv.inRange(img_array, 2, 2)
    ret_wall, thresh_wall = cv.threshold(wall_mask, 2, 1, cv.THRESH_BINARY)
    # Maska komory
    ventricle_mask = cv.inRange(img_array, 1, 1)
    ret_ven, thresh_ven = cv.threshold(ventricle_mask, 1, 1, cv.THRESH_BINARY)
    # LVAnalysis krawędziowy ściany
    edged_wall = cv.Canny(thresh_wall, 0, 1)
    edged_wall_c = cv.cvtColor(edged_wall, cv.COLOR_GRAY2BGR)

    # Linia endocardialna poprzez dylatację maski komory i koniunkcję z obrazem krawędziownym ściany
    kernel = np.ones((3, 3), np.uint8)
    dilated_ven = cv.dilate(thresh_ven, kernel, iterations=1)  # dilate ventricle mask
    endocardial_image = cv.bitwise_and(edged_wall, dilated_ven)

    # Linia zewnętrzna
    dilated_ven2 = cv.dilate(dilated_ven, kernel, iterations=1)
    dilated_ven2[dilated_ven2 > 0] = 255
    not_dil_ven = cv.bitwise_not(dilated_ven2)
    wall_outer_line = np.zeros(edged_wall.shape, np.uint8)
    cv.bitwise_and(edged_wall, not_dil_ven, wall_outer_line)
    contours_outer, hierarchy_outer = cv.findContours(wall_outer_line, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_wall, hierarchy_wall = cv.findContours(thresh_wall, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Zewnętrzna linia - szukanie linii prostych
    # Zamknięcie maski ściany w wielokącie
    hull_img = np.zeros((img_array.shape[0], img_array.shape[1], 3), np.uint8)
    hull_l = []
    for i in range(len(contours_outer)):
        hull_l.append(cv.convexHull(contours_outer[i], False))
    cv.drawContours(hull_img, hull_l, -1, (0, 255, 0), 1)
    hull_img_g = cv.cvtColor(hull_img, cv.COLOR_BGR2GRAY)

    # Wykrywanie linii transformatą Hougha
    lines = cv.HoughLines(hull_img_g, 1, np.pi / 180, 60, None, 0, 0)

    # Środek ciękości ściany
    cY = None
    for cnt in contours_outer:
        M = cv.moments(cnt)
        cY = int(M["m01"] / (M["m00"] + 1e-5))

    x = np.arange(0, img_array.shape[1])

    def line_equation(x):
        x1 = pt2[0]
        y1 = pt2[1]
        x2 = pt1[0]
        y2 = pt1[1]
        a = (y2 - y1) / (x2 - x1)
        return a * (x - x1) + y1

    # Wykrywanie linii stanowiących podstawy ściany komory
    det_lines = []
    polygons = []
    if lines is not None:
        for i in range(0, len(lines)):
            # Wyszukiwanie linii prostych wraz z ich równaniami
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Wybór 2 punktów na prostej
            pt1 = (int(x0 + 50 * (-b)), int(y0 + 50 * (a)))
            pt2 = (int(x0 - 50 * (-b)), int(y0 - 50 * (a)))
            # Wyszukiwanie kąta nachylenia prostej względem obrazu
            angle = np.arctan2(pt1[1] - pt2[1], pt2[0] - pt1[0]) * 180.0 / np.pi
            # Filtracja linii wertykalnych mieszczących się w pewnym zakresie kątów nachylenia
            if 57.0 > angle > -57.0:
                # Tworzenie wektora linii na podstawie jej równania
                line = np.vectorize(line_equation)
                y = line(x).astype(np.uint)
                # Wyznaczenie punktu stanowiącego środek linii
                pt1 = (x[0], y[0])
                pt2 = (x[-1], y[-1])
                line_mid = (x[int(len(x) / 2)], y[int(len(y) / 2)])
                # Jeżeli środek linii znajduje się poniżej środka ciężkości komory
                if line_mid[1] > cY:
                    # Storzenie czworokąta, w którego obszarze mieści się dana linia
                    cv.line(edged_wall_c, (pt1[0], pt1[1] - 8), (pt2[0], pt2[1] - 8), (0, 255, 255), 1, cv.LINE_AA)
                    cv.line(edged_wall_c, (pt1[0], pt1[1] + 10), (pt2[0], pt2[1] + 10), (0, 255, 255), 1, cv.LINE_AA)
                    # Zapise czworokąta do listy wielomianów
                    polygons.append(Polygon(
                        [(pt1[0], pt1[1] - 8), (pt1[0], pt1[1] + 10), (pt2[0], pt2[1] + 10), (pt2[0], pt2[1] - 8)]))
                    # Zapis punktów końcowych linii do listy
                    det_lines.append((pt1, pt2))

    acute_ang = []
    angles = []
    approx = []

    # Aproksymacja konturu ściany
    for i in range(len(contours_wall)):
        epsilon = 0.0008 * cv.arcLength(contours_wall[i], True)
        approx.append(cv.approxPolyDP(contours_wall[i], epsilon, True))

    cv.drawContours(edged_wall_c, approx, -1, (0, 255, 0), 1)

    # Wyszukiwanie ostrych kątów w aproksymowanej ściane, w punktach mieszczących się w pewnym ograniczonym obszarze
    for app in approx:  # Dla każdego konturu app w konturach approx
        for i in range(len(app)):
            # Obliczanie kąta na podstawie 3 punktów w aproksymowanym konturze
            pt1 = app[i - 1][0]
            pt2 = app[i][0]
            # Sprawdzanie czy następny indeks nie wskazuje na ostatni punkt konturze app
            if i != (len(app) - 1):
                pt3 = app[i + 1][0]  # Jeżeli nie jest zapisywany jest jako 3. punkt (pt3)
            else:
                pt3 = app[0][0]
            v1 = vector(pt1, pt2)  # Obliczanie wektora między punktem pt1 a pt2
            v2 = vector(pt2, pt3)  # Obliczanie wektora między punktem pt2 a pt3
            if v1 is not None and v2 is not None:  # Jeżeli wektor został obliczony
                mag1 = magnitude(v1)  # Obliczanie długości obu wektorów
                mag2 = magnitude(v2)
                dot = dot_product(v1, v2)  # Obliczanie iloczynu skalarnego wektorów
                ang = calculate_angle(mag1, mag2, dot) * 180.0 / np.pi  # Obliczanie kąta pomiędzy wektorami
                if ang >= 21:  # Jeżeli kąt jest większy lub równy 21 stopni
                    point = Point(pt2[0], pt2[1])  # Zapis punktu p2 do zmiennej point
                    for poly in polygons:  # Dla każego wielokąta w liście polygons
                        if poly.contains(
                                point):  # Jeżeli punkt znajduje się w obrzarze wielokąta, dodawany jest do listy
                            # punktów z wierzchołkami kątów ostrych - acute_ang
                            acute_ang.append(tuple(pt2))
                            break
                angles.append(ang)
    acute_ang.sort(key=lambda x: x[0])  # Sortowanie punktów wg współrzędnych x rosnących

    # Filtracja kątów ostrych
    del_idx = []
    i = 0
    while i < len(acute_ang) - 1:
        if distance(acute_ang[i], acute_ang[i + 1]) < 25:  # Jeżeli odległości y między punktami jest mniejsza niż 25 px
            if acute_ang[i + 1][1] < acute_ang[i][1]:  # Usuwanie punktu położonego wyżej
                del_idx.append(i + 1)
            else:
                del_idx.append(i)
        i = i + 1
    # Punkty po filtracji oznaczające początki i końce podstaw ściany
    det_points = [ang for i, ang in enumerate(acute_ang) if i not in del_idx]

    # Wyznaczenie linii epicardialnej
    epicardial_image = np.zeros(edged_wall.shape, np.uint8)
    temp = np.zeros(edged_wall.shape, np.uint8)  # LVAnalysis krawędziowy ściany bez linii endocardialnej
    # Podstawa ściany z lewej strony
    x = np.arange(det_points[0][0], det_points[1][0])  # Tworzenie wektora pomiędzy punktami krańcowymi podstawy
    line_v = np.vectorize(line_vector)
    y = line_v(x, det_points[0][0], det_points[1][0] + 1, det_points[0][1], det_points[1][1] + 1).astype(np.uint)
    # pt1 pt2 - nowe punkty krańców podstawy w celu uniknięcia przycięcia linii epicardialnej
    pt1 = (x[3], y[3])
    pt2 = (x[-1], y[-1])
    # Podstawa ściany z prawej strony - analogicznie jak dla lewej podstawy
    x = np.arange(det_points[2][0], det_points[3][0])
    line_v = np.vectorize(line_vector)
    y = line_v(x, det_points[2][0], det_points[3][0], det_points[2][1], det_points[3][1]).astype(np.uint)
    pt3 = (x[0], y[0])
    pt4 = (x[-3], y[-3])
    # Usunięcie podstaw ściany
    cv.line(temp, pt1, pt2, 255, 5, cv.LINE_AA)  # Tworzenie linii podstaw o grubości 5 px
    cv.line(temp, pt3, pt4, 255, 5, cv.LINE_AA)
    mid_end1 = midpoint(pt1, pt2)
    mid_end2 = midpoint(pt3, pt4)
    temp[temp > 0] = 255  # Zamiana obrazu temp na obraz binarny 8-bitowy
    cv.bitwise_and(wall_outer_line, cv.bitwise_not(temp),
                   epicardial_image)  # Usuwanie podstaw z obrazu krawędziowego, poprzez negację
    endocardial_image[endocardial_image > 0] = 255

    # Wykrywanie konturu epi oraz endo jako zbiór białych pixeli wynaczających linie
    cnt_endo = np.argwhere(endocardial_image == 255)
    cnt_epi = np.argwhere(epicardial_image == 255)

    # Punkty konturów - zamiana miejsc x i y współrzędnych konturu
    x = list()
    y = list()
    x.extend(list(map(int, [pt[1] for pt in cnt_endo])))
    y.extend(list(map(int, [pt[0] for pt in cnt_endo])))
    points_endo = list(zip(x, y))
    x.clear()
    y.clear()
    x.extend(list(map(int, [pt[1] for pt in cnt_epi])))
    y.extend(list(map(int, [pt[0] for pt in cnt_epi])))
    points_epi = list(zip(x, y))

    # Wyznaczenie środka ciężkości konturu komory za pomocą linii endocardialnej
    sum_x = 0
    sum_y = 0
    for pt in points_endo:
        sum_x = sum_x + pt[0]
        sum_y = sum_y + pt[1]

    mid_x = sum_x / len(points_endo)
    mid_y = sum_y / len(points_endo)

    # Zamiana współrzędnych linii endocardialnej z kartejańskich na biegunowe
    points_endo_polar = []
    for pt in points_endo:
        r = math.sqrt((pt[0] - mid_x) ** 2 + (pt[1] - mid_y) ** 2)
        theta = math.atan2(pt[1] - mid_y, pt[0] - mid_x)
        points_endo_polar.append((r, theta))

    # Sortowanie punktów linii endocardialnej w układzie biegunowym
    tresh = None
    points_endo_polar.sort(key=lambda x: x[1])
    for i in range(0, len(points_endo_polar) - 1):
        # Różnica kątów pomiędzy kolejnymi punktami
        diff = abs(points_endo_polar[i][1] - points_endo_polar[i + 1][1])
        if math.pi > diff > 0.1:  # Jeżeli większa od 0.1 radiana to punkty leżą po przeciwległych krańcach podstawy
            tresh = points_endo_polar[i][1] + diff / 2  # Tresh - granica znajdująca się na podstawie

    # Podział linii na względem kąta progowego - tresh
    a = [x for x in points_endo_polar if x[1] > tresh]
    b = [x for x in points_endo_polar if x[1] < tresh]
    c = a + b
    points_endo_polar.clear()
    points_endo_polar = c[:]  # Przypisanie zmiennej @points_endo_polar współrzędnych o zmienionej kolejności

    # Linia endocardialna - zmienne biegunowe na kartezjańskie
    endo_points = []
    for i, pt in enumerate(points_endo_polar):
        x = int(mid_x + pt[0] * math.cos(pt[1]))
        y = int(mid_y + pt[0] * math.sin(pt[1]))
        endo_points.append([x, y])

    # Obliczanie długość lini endocardialnej
    endo_length = wall_lines_length(endo_points, 0, len(endo_points) - 1, 20)

    # Zamiana współrzędnych linii epicardialnej z kartejańskich na biegunowe
    points_epi_polar = []
    for pt in points_epi:
        r = math.sqrt((pt[0] - mid_x) ** 2 + (pt[1] - mid_y) ** 2)
        theta = math.atan2(pt[1] - mid_y, pt[0] - mid_x)
        points_epi_polar.append((r, theta))

    # Sortowanie punktów linii epicardialnej w układzie biegunowym
    points_epi_polar.sort(key=lambda x: x[1])
    for i in range(0, len(points_epi_polar) - 1):
        diff = abs(points_epi_polar[i][1] - points_epi_polar[i + 1][1])
        if math.pi > diff > 0.2:
            tresh = points_epi_polar[i][1] + diff / 2

    a = [x for x in points_epi_polar if x[1] > tresh]
    b = [x for x in points_epi_polar if x[1] < tresh]
    c = a + b
    points_epi_polar.clear()
    points_epi_polar = c[:]

    # Linia epicardialna - zmienne biegunowe na kartezjańskie
    epi_points = []
    for i, pt in enumerate(points_epi_polar):
        x = int(mid_x + pt[0] * math.cos(pt[1]))
        y = int(mid_y + pt[0] * math.sin(pt[1]))
        epi_points.append([x, y])

    # Obliczanie długości lini epicardialnej
    epi_length = wall_lines_length(epi_points, 0, len(epi_points) - 1, 20)

    # Wyznaczenie linii środkowej

    # Wyznaczenie lini wzdłuż konturu
    epi_lines = []
    for i in range(0, len(epi_points) - 5, 5):
        epi_lines.append([epi_points[i], epi_points[i + 5]])

    # Linie ortogonalne do linii endocardialnej
    ort_lines = []
    mid_points = []
    line_v = np.vectorize(ortogonal_vector)
    line_vec = np.vectorize(line_vector)

    for i in range(0, len(endo_points) - 4, 4):
        # x0, y0 - środek odcinka w linii endocardialnej,
        # a - nachylenie linii ortogonalnej przechodzącej przez punkt (x0,y0)
        a, x0, y0 = ortogonal(endo_points[i], endo_points[i + 4])
        if a is None:
            y = np.arange(y0 - 140, y0 + 40)  # Wektor linii ortagonalnej
            x = line_v(y, a, x0, y0).astype(np.int64)
        if a is not None:
            x = np.arange(x0 - 140, x0 + 140)
            y = line_v(x, a, x0, y0).astype(np.int64)
        # Ograniczenie linii do wymiarów obrazu
        nx = []
        ny = []
        for i in range(len(x)):
            if img_array.shape[1] > x[i] > 0 and 0 < y[i] < img_array.shape[0]:
                nx.append(x[i])
                ny.append(y[i])
        # Dodanie punktu początkowego oraz końcowego linii ortogonalnej do zmiennej oline
        oline = [[nx[0], ny[0]], [nx[-1], ny[-1]]]
        ort_lines.append(oline)
        inter_pt = []
        det_inter_pt = []
        for eline in epi_lines:
            pt = line_intersect(eline, oline)  # Punkt przecięcia linii ortogonalnej z linią epicardialną
            inter_pt.clear()
            if pt is not None:  # Jeżeli wykryto punkt przecięcia
                if distance(pt, (x0, y0)) < 140:
                    # Punkt akceptowalny (dodawany do listy inter_pt)
                    # jeżeli odległość od linii endo do epi jest mniejsza od 150 (graniczna wartość)
                    inter_pt.append(pt)
                    for pt in inter_pt:
                        # Tworzenie wektora między punktami na linii endo i epi, w zależnośći od położenia punktów
                        if pt[0] < x0:
                            xv = np.arange(pt[0], x0).astype(np.int64)
                            yv = line_vec(xv, pt[0], x0, pt[1], y0).astype(np.int64)
                        elif pt[0] > x0:
                            xv = np.arange(x0, pt[0]).astype(np.int64)
                            yv = line_vec(xv, x0, pt[0], y0, pt[1]).astype(np.int64)
                        else:
                            # Gdy punkt endo oraz epi znajdują się na linii poziomej (ta sama współrzędna x)
                            yv = np.arange(y0, pt[1]).astype(np.int64)
                            xv = np.full((1, len(yv)), x0).astype(np.int64)
                            # Zliczanie w wektorze ilość pixeli należących do obszaru komory
                        count = 0
                        for k in range(0, len(xv) - 1):
                            if img_array[yv[k]][xv[k]] == 1:
                                count = count + 1
                        if count < 3:  # Jeżeli ilość pixeli komory jest mniejsza od 5 to punkt jest akceptowany
                            tmp_pt = pt
                            inter_pt.clear()
                            det_inter_pt.append(tmp_pt)  # Zapis punktu do listy zaakceptowanych punktów

        if det_inter_pt:  # Jeżeli lista puntów zaakceptowanych nie jest pusta
            pt = det_inter_pt.pop(0)  # Ostatni punkt na liście jako punkt linii epicardialnej
            mid_pt = midpoint(pt, (x0, y0))  # Środek linii łączący punkty epi i endo jako punkt linii środkowej
            mid_points.append(mid_pt)

    # Linia środkowa (myocardial) - zmiana współrzędnych kartezjańskich na biegunowe
    points_mid_polar = []
    for pt in mid_points:
        r = math.sqrt((pt[0] - mid_x) ** 2 + (pt[1] - mid_y) ** 2)
        theta = math.atan2(pt[1] - mid_y, pt[0] - mid_x)
        points_mid_polar.append((r, theta))

    # Sortowanie współrzędnych biegunowych linii środkowej
    points_mid_polar.sort(key=lambda x: x[1])
    a = [x for x in points_mid_polar if x[1] > tresh]
    b = [x for x in points_mid_polar if x[1] < tresh]
    c = a + b
    points_mid_polar.clear()
    points_mid_polar = c[:]
    mid_points.clear()

    # Zamiana współrzędnych biegunowych na kartezjańskie linii środkowej
    for i, pt in enumerate(points_mid_polar):
        x = int(mid_x + pt[0] * math.cos(pt[1]))
        y = int(mid_y + pt[0] * math.sin(pt[1]))
        mid_points.append([x, y])
    mid_points.insert(0, mid_end1)  # Dodanie do linii 1. punktu (środek lewej podstawy ściany)
    mid_points.append(mid_end2)  # Dodanie do linii ostatniego punktu (środek prawej podstawy ściany)

    # Wygładzenie i usuwanie szumu krzywej środkowej poprzez filtr Savitzky-Golay
    x = []
    y = []
    for i in range(0, len(mid_points)):
        x.append(mid_points[i][0])
        y.append(mid_points[i][1])

    # Filtracja 1.
    y_filtered_1 = signal.savgol_filter(y, 7, 5)  # rozmiar maski: 7, stopień wielomianu:5
    x_filtered_1 = signal.savgol_filter(x, 7, 3)

    # Filtracja 2.
    y_filtered_2 = signal.savgol_filter(y_filtered_1, 9, 5)
    x_filtered_2 = signal.savgol_filter(x_filtered_1, 5, 2)

    # Połączenie współrzędnych x, y i zapis do listy
    mid_points = []
    for i in range(0, len(x_filtered_2) - 1):
        mid_points.append([int(x_filtered_2[i]), int(y_filtered_2[i])])

    # Uzupełnienie konturu o oryginalny punkt początkowy i końcowy konturu (w przypadku skrócenia linii podczas
    # filtracji)
    mid_points.append([x[-1], y[-1]])
    mid_points.insert(0, [x[0], y[0]])

    # Obliczenie długości lini środkowej
    mid_length = wall_lines_length(mid_points, 0, len(mid_points) - 1, 3)

    # Aproksymacja linii środkowej
    mid_cnt_array = np.array([mid_points])
    mid_cnt_app = []
    for i in range(len(mid_cnt_array)):
        mid_cnt_app.append(cv.approxPolyDP(mid_cnt_array[i], 1, closed=False))

    # Rysowanie konturów linii
    contours_epi, _ = cv.findContours(epicardial_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours_endo, _ = cv.findContours(endocardial_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for img_out in images_out:
        for cnt in contours_epi:
            cv.drawContours(img_out, cnt, -1, (255, 0, 0), 2)
        for cnt in contours_endo:
            cv.drawContours(img_out, cnt, -1, (0, 0, 255), 2)
        cv.polylines(img_out, mid_cnt_app, False, (200, 0, 200), 2)

    return images_out, epi_points, mid_points, endo_points, round(epi_length, 2), round(mid_length, 2), round(
        endo_length, 2)
