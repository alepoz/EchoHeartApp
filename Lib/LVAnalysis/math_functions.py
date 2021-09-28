import math


def magnitude(v):
    """
    Oblicza długość wektora v
    :param v: wektor
    :type v: list
    :return: mag: długość wektora
    :rtype: float
    """
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2)
    return mag


def vector(pt1, pt2):
    """
    Oblicza wektor pomiędzy punktami pt1 i pt2
    :param pt1: punkt 1. (x,y)
    :type pt1: tuple
    :param pt2: punkt 2. (x,y)
    :type pt2: tuple
    :return: vector: wektor
    :rtype: list
    """
    vector = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
    if vector[0] == 0 and vector[1] == 0:
        return None
    else:
        return vector


def dot_product(vector1, vector2):
    """
    Oblicza iloczyn skalarny dwóch wektorów
    :param vector1: wektor 1.
    :type vector1: list
    :param vector2: wektor 2.
    :type vector2: list
    :return: dot: iloczyn skalarny
    :rtype: float
    """
    dot = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    return dot


def calculate_angle(mag1, mag2, dot):
    """
    Funkcja obliczająca kąt pomiędzy dwoma wektorami
    :param mag1: długość 1. wektora
    :type mag1: float
    :param mag2: długość 2. wektora
    :type: float
    :param dot: iloczyn skalarny
    :type dot: float
    :return: kąt
    :rtype: float
    """
    cosinus = dot / (mag1 * mag2)
    cosinus = round(cosinus, 2)
    if 1 >= cosinus >= -1:
        return math.acos(cosinus)


def distance(pt1, pt2):
    """
    Oblicza odległość euklidesową pomiędzy punktami: pt1, pt2 w jednostkach pikseli
    :param pt1: punkt 1. (x,y)
    :param pt2: punkt 2. (x,y)
    :return: dystans
    :rtype: float
    """
    dist = math.sqrt(((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2))
    return dist


def distance_cm(pt1, pt2):
    """
       Oblicza rzeczywistą odległość euklidesową pomiędzy punktami: pt1, pt2 w jednostkach centymetra
       :param pt1: punkt 1. (x,y)
       :param pt2: punkt 2. (x,y)
       :return: dystans  w cm
       :rtype: float
       """
    pt1 = list(pt1)
    pt2 = list(pt2)
    pt1[0] *= 0.0308
    pt2[0] *= 0.0308
    pt1[1] *= 0.0154
    pt2[1] *= 0.0154
    dist_cm = math.sqrt(((pt2[0] - pt1[0]) ** 2) + ((pt2[1] - pt1[1]) ** 2))
    return dist_cm


def line_vector(x, x1, x2, y1, y2):
    """
        Oblicza y dla każdego elementu z tablicy x, wg równania linii przechodzącej przez punkty: (x1,y1), (x2, y2)
        :param x: tablica współrzędnych x
        :type x: numpy.array
        :param x1: współrzędna x 1. punktu, przez który przechodzi prosta
        :type x1: tuple/list
        :param y1: współrzędna y 1. punktu, przez który przechodzi prosta
        :type x2: tuple/list
        :param x1: współrzędna x 2. punktu, przez który przechodzi prosta
        :type x2: tuple/list
        :param y1: współrzędna y 2. punktu, przez który przechodzi prosta
        :type y2: tuple/list
        :return: tablica współrzędnych y
        :rtype: numpy.array
        """
    a = (y2 - y1) / (x2 - x1)
    return a * (x - x1) + y1


def ortogonal(pt1, pt2):
    """
    Oblicza punkt będący środkiem odległości między punktami: pt1 i pt2 oraz współczynnik nachylenia prostej
    ortogonalnej do ddcinka pt1pt2
    :param pt1: punkt 1. (x,y)
    :type pt1: tuple/list
    :param pt2: punkt 2. (x,y)
    :type pt2: tuple/list
    :return: współczynnik nachylenia, współrzędne punktu środkowego
    :rtype: float, int, int
    """
    pt = midpoint(pt1, pt2)
    # y taki sam (linia pozioma)
    if (pt2[1] - pt1[1]) == 0:
        # a wskazuje na linię pionową
        a = None
    # x taki sam (linia pionowa)
    elif (pt2[0] - pt1[0]) == 0:
        # a wskazuje na linię poziomą
        a = 0
    else:
        a = -1 / ((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))
    return a, pt[0], pt[1]


def ortogonal_vector(x, a, x1, y1):
    """
    Oblicza y dla każdego elementu z tablicy x, wg równania linii o współczynniku nachylenia a i
    przechodzącej przez punkt (x1,y1)
    :param x: tablica współrzędnych x
    :type x: numpy.array
    :param a: współczynnik nachylenia linii
    :type a: float
    :param x1: współrzędna x punktu, przez który przechodzi prosta
    :type x1: tuple/list
    :param y1: współrzędna y punktu, przez który przechodzi prosta
    :type y1: tuple/list
    :return: tablica współrzędnych y
    :rtype: numpy.array
    """
    if a is not None:
        return a * (x - x1) + y1
    if a is None:
        return x1


def midpoint(pt1, pt2):
    """
    Oblicza punkt stanowiący środek odległości pomiędzy punktami pt1 i pt2
    :param pt1: punkt 1. (x,y)
    :type pt1: tuple/list
    :param pt2: punkt 2. (x,y)
    :return: mid_pt: punkt środkowy pomiędzy pt1, pt2
    :rtype: tuple
    """
    mid_pt = int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)
    return mid_pt


def line_intersect(l1, l2):
    """
    Poszukuje punkt przecięcia się 2 linii
    :param l1: linia 1. ze współrzędnymi: punkt początkowy (x,y) i końcowy (x,y)
    :type l1: list
    :param l2: linia 2. ze współrzędnymi: punkt początkowy (x,y) i końcowy (x,y)
    :type l2: list
    :return: x,y: współrzędne punktu
    :rtype: float, float
    """
    # Linia A
    # punkt 1.
    ax1 = int(l1[0][0])
    ay1 = int(l1[0][1])
    # punkt 2.
    ax2 = int(l1[1][0])
    ay2 = int(l1[1][1])

    # Linia B
    # punkt 1.
    bx1 = int(l2[0][0])
    by1 = int(l2[0][1])
    # punkt 2.
    bx2 = int(l2[1][0])
    by2 = int(l2[1][1])

    d = (ax2 - ax1) * (by2 - by1) - (bx2 - bx1) * (ay2 - ay1)

    if d:
        # Linia A
        t = ((ax1 - bx1) * (by1 - by2) - (ay1 - by1) * (bx1 - bx2)) / d
        # Linia B
        u = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d

    else:
        return
    if not (0 <= t <= 1 and 0 <= u <= 1):
        return
    x = ax1 + t * (ax2 - ax1)
    y = ay1 + t * (ay2 - ay1)

    return x, y
