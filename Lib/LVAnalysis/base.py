import cv2.cv2 as cv
import numpy as np
from shapely.geometry import Point, Polygon
from Lib.LVAnalysis.math_functions import *


def detect_base_2(img_array):
    """
    Wyznacza położenie 3 punktów podstawy
    :param img_array: obraz maski
    :type img_array: numpy.array
    :return: mid_base: punkt środkowy, pt_1: 1. punkt pt_2: 2. punkt
    :rtype: tuple, tuple, tuple
    """
    ventricle_mask = cv.inRange(img_array, 1, 1)  # Obszar komory
    ret_ven, thresh_ven = cv.threshold(ventricle_mask, 1, 1, cv.THRESH_BINARY)  # Binarna maska komory
    contours_ven, hierarchy_ven = cv.findContours(thresh_ven, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Kontur komory
    for cnt in contours_ven:
        M = cv.moments(cnt)
        cY = int(M["m01"] / (M["m00"] + 1e-5))  # Współrzędna y środka ciężkości komory

    edged_ven = cv.Canny(thresh_ven, 0, 1)  # Obraz krawędziowy komory
    x = np.arange(0, img_array.shape[1])  # Wektor współrzędnych x obrazu

    def line_equation(x):  # Równanie prostej przechodzącej przez 2 punkty
        x1 = pt2[0]
        y1 = pt2[1]
        x2 = pt1[0]
        y2 = pt1[1]
        a = (y2 - y1) / (x2 - x1)  # Współczynnik nachylenia prostej
        return a * (x - x1) + y1  # Wpółrzędna y obliczona dla x

    polygons = []  # Lista przechowująca wielokąty
    det_lines = []  # Lista wykrytych linii
    lines = cv.HoughLines(edged_ven, 1, np.pi / 180, 40, None, 0, 0)  # Wykrywanie linii transformatą Hougha
    if lines is not None:  # Jeżeli wykryto linie
        for i in range(0, len(lines)):  # Dla każdej linii
            r = lines[i][0][0]  # Promień
            theta = lines[i][0][1]  # Kąt
            a = math.cos(theta)  # Cosinus kata theta
            b = math.sin(theta)  # Sinus kata theta
            # x0, y0 - współrzędne końca linii
            x0 = a * r
            y0 = b * r
            # pt1, pt2 - 2 punkty znajdujące się na linii
            pt1 = (int(x0 + 50 * (-b)), int(y0 + 50 * a))
            pt2 = (int(x0 - 50 * (-b)), int(y0 - 50 * a))
            # Kąt nachylenia linii
            angle = np.arctan2(pt1[1] - pt2[1], pt2[0] - pt1[0]) * 180.0 / np.pi
            if 55.0 > angle > -55.0:  # Jeżeli kąt mieści się w przedziale (-55, 55) stopni
                line = np.vectorize(line_equation)  # Wektoryzacja funkcji line_equation
                y = line(x).astype(np.uint)  # Oliczenie wpół. y dla każdej współ x obrazu wg równania linii
                pt1 = (x[0], y[0])  # Współrzędne początku linii
                pt2 = (x[-1], y[-1]) # Współrzędne końca linii
                line_mid = (x[int(len(x) / 2)], y[int(len(y) / 2)])  # Środek linii
                if line_mid[1] > cY:  # Jeżeli środek linii znajdye się poniżej środka ciężkości linii
                    # Utwórz równoległobok o podstawach tworzonych przez linię przesuniętą o 10 pikseli do dołu oraz
                    # przesuniętą o 8 pikseli do góry, dodaj go do listy wielokątów
                    polygons.append(Polygon(
                        [(pt1[0], pt1[1] - 8), (pt1[0], pt1[1] + 10), (pt2[0], pt2[1] + 10), (pt2[0], pt2[1] - 8)]))
                    det_lines.append((pt1, pt2))  # Zapisz współrzędne początka i końca linii

    approx = [] # Kontur aproksymowany
    for i in range(len(contours_ven)):
        # Dobierz epsilon aproksymacji jako 0.002 długości konturu
        epsilon = 0.002 * cv.arcLength(contours_ven[i], True)
        approx.append(cv.approxPolyDP(contours_ven[i], epsilon, True))  # Aproksymuj kontur komory

    acute_pts = []  # Lista punktów wykrytych wierzchołków kątów
    angles = []  # Lista wykrytych kątów
    for app in approx: # DLa każdego aproksymowanego konturu w liście aproksymowanych
        for i in range(len(app)):  # Dla każdego punktu w aproksymowanym konturze
            pt1 = app[i - 1][0]  # Pierwszy punkt
            pt2 = app[i][0]  # Drugi punkt
            if i != (len(app) - 1):  # Jeżeli 2. punkt nie jest ostatnim w konturze
                pt3 = app[i + 1][0]  # Trzeci punkt
            else:
                pt3 = app[0][0]  # Trzeci punkt
            v1 = vector(pt1, pt2)  # Utworzenie wektora pomiędzy punktem 1. a 2.
            v2 = vector(pt2, pt3)  # Utworzenie wektora pomiędzy punktem 2. a 3.
            if v1 is not None and v2 is not None:  # Jeżeli wektory zostały policzone
                mag1 = magnitude(v1)  # Długość wektora 1.
                mag2 = magnitude(v2)  # Długość wektora 2.
                dot = dot_product(v1, v2)  # Iloczyn skalarny wektora 1. i 2.
                ang = calculate_angle(mag1, mag2, dot) * 180.0 / np.pi  # Oblicz kąt pomiędzy wektorami
                if ang >= 19:  # Jeżeli kąt jest większy od 19 stopni
                    point = Point(pt2[0], pt2[1])  # Zapisz współrzędne punktu 2. jako obiekt Point
                    for poly in polygons:  # sprawdź czy w środku któregokolwiek równoległoboku znajduje się ten punkt
                        if poly.contains(point):  # Jeżeli tak
                            acute_pts.append(tuple(pt2))  # Zapisz punkt 2. do listy
                            angles.append(ang)  # Zapisz kąt do listy
                            break

    if len(acute_pts) != 2:  # Jeżeli ilość wykrytych punktów nie jest równa 2
        acute_pts.sort(key=lambda x: x[0])  # Sortuj punkty wg współ. x w kolejności rosnącej
        del_idx = []
        i = 0
        while i < len(acute_pts) - 1:
            # Jeżeli dystans pomiędzy 2 kolejnymi punktami jest mniejszy niż 25
            if distance(acute_pts[i], acute_pts[i + 1]) < 25:
                if acute_pts[i + 1][1] < acute_pts[i][1]:
                    del_idx.append(i + 1)  # Dodaj indeks punktu o wyższej wsółrzędnej y do listy indeksów
                else:
                    del_idx.append(i)
            i = i + 1
        # Usuń punkty o indeksach zapisanych w liście indeksów
        det_points = [j for i, j in enumerate(acute_pts) if i not in del_idx]
        acute_pts = det_points.copy()

    mid_base = None
    pt_1 = None
    pt_2 = None
    if len(acute_pts) == 2:  # Jeżeli ilość punktów po filtracji jest równa 2
        # Zapisz współrzędne tych punktów jako końce podstawy
        pt_1 = (int(acute_pts[0][0]), int(acute_pts[0][1]))
        pt_2 = (int(acute_pts[1][0]), int(acute_pts[1][1]))
        mid_base = midpoint(pt_1, pt_2)  # Oblicz środek podstawy
    else:
        print(f"Nie poprawna ilość punktów: {len(acute_pts)}")

    return mid_base, pt_1, pt_2


def detect_base(img_array, images_out):
    """
       Wyznacza położenie 3 punktów podstawy oraz jej długość
       :param img_array: obraz maski
       :type img_array: numpy.array
       :param images_out: obrazy wyjściowe
       :type images_out: list
       :return: obrazy wyjściowe, punkty podstawy: środkowy, lewy, prawy, długość podstawy
       :rtype: list, tuple, tuple, tuple, float
       """
    atr_mask = cv.inRange(img_array, 3, 3)
    det_1px = None
    det_2px = None

    if cv.countNonZero(atr_mask) > 40:  # Czy obszar przedionka ma ponad 40 pikseli
        prev = list()  # Lista przechowująca wartości pikseli poprzedniej linii
        for y in range(img_array.shape[0] - 1, 1, -1):
            if not det_1px:  # Jeżeli 1. punkt końca odcinka podstawy nie został znaleziony
                for x in range(1, img_array.shape[1]):
                    # Jeżeli piksel o współrzędnych (x,y) należy do komory, a piksel pod nim do przedsionka
                    if img_array[y][x] == 1 and img_array[y + 1][x] == 3:
                        det_1px = (x, y)  # Zapisz współrzędne piksela jako 1. punkt podstawy
                        line = list(img_array[y])  # Zapis wartości pikseli aktualnej linii obrazu
                        is_det_atr = line.count(3)  # Ilość pikseli należących do przedsionka w linii
                        if is_det_atr == 0:  # Jeżeli ilość wynosi 0
                            # Wyszukaj wszystkie współ. x pikseli należacych do komory w tej linii
                            det_ven = [i for i, n in enumerate(line) if n == 1]
                            # Przypisz 2. punktowi podstawy współrzędne ostatniego piksela komory w linii
                            det_2px = (det_ven[-1], y)
                            break  # Przerwij pętlę
                        else:
                            prev = line[:]  # Zapisz piksele aktualnej linii jako linia poprzednia
                            break
            else:
                if det_2px:  # Jeżeli 2. punkt końca odcinka podstawy został znaleziony
                    break  # Przerwij pętlę
                else:
                    line = list(img_array[y])  # Zapis wartości pikseli aktualnej linii obrazu
                    is_det_atr = line.count(3)  # Ilość pikseli należących do przedsionka w linii
                    if is_det_atr == 0:  # Jeżeli ilość wynosi 0
                        # Wyszukaj wszystkie współ. x pikseli należacych do przedsionka w poprzedniej linii
                        det_atr = [i for i, n in enumerate(prev) if n == 3]
                        px_atr = min(det_atr)  # Najmniejsza współrzędna x
                        if px_atr < det_1px[0]:  # Jeżeli 1. punkt jest położony na prawo od współ. px_atr
                            # Przypisz 2. punktowi podstawy współ. x pierwszego piksela przedsionka z poprzedniej linii
                            # oraz współ. y aktualnej linii
                            det_2px = (det_atr[0], y)
                            break
                        else: # Jeżeli 1. punkt jest położony na lewo od współ. px_atr
                            # Przypisz 2. punktowi podstawy współ. x ostatniego piksela przedsionka z poprzedniej linii
                            # oraz współ. y aktualnej linii
                            det_2px = (det_atr[-1], y)
                            break
                    else:
                        prev = line[:]  # Zapisz piksele aktualnej linii jako linia poprzednia
                        continue
        # Jeżeli 1. wykryty punkt końca znajduje się na prawo od 2. punktu podstawy
        if det_2px[0] < det_1px[0]:
            y = det_1px[1]  # Zapisz współ. y punktu 1.
            for x in range(det_1px[0] + 1, img_array.shape[1]):
                # W linii obrazu punktu 1. sprawdź dla kolejnych pikseli na prawo czy spełniają założenia: piksel
                # znajduje się w obszarze komory, a piksel pod nim w obszarze przedsionka
                if img_array[y][x] == 1 and img_array[y + 1][x] == 3:
                    det_1px = (x, y) # Jeżeli tak, zapisz do punktu nowe współrzędne
                else:
                    break

    if cv.countNonZero(atr_mask) < 40:  # Jeżeli obszar przedsionka jest mniejszy niż 40 px
        base_mid, det_1px, det_2px = detect_base_2(img_array)

    mid_base = midpoint(det_1px, det_2px)  # Oblicz punkt środka odcinka podstawy

    base_length = round(distance_cm(det_1px, det_2px), 2)  # Oblicz długość podstawy

    if det_1px[0] < det_2px[0]:  # Zdefiniuj punkty podstawy jako prawy i lewy
        left_base = det_1px
        right_base = det_2px
    else:
        left_base = det_2px
        right_base = det_1px

    # Rysuj na obrazach: odcinek podstawy oraz 3 punkty
    for img_out in images_out:
        cv.line(img_out, det_1px, det_2px, (0, 120, 255), 2, cv.LINE_AA)
        cv.circle(img_out, det_1px, radius=3, color=(0, 255, 0), thickness=-1)
        cv.circle(img_out, mid_base, radius=3, color=(0, 255, 0), thickness=-1)
        cv.circle(img_out, det_2px, radius=3, color=(0, 255, 0), thickness=-1)

    return images_out, mid_base, left_base, right_base, base_length
