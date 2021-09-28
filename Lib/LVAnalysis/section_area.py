import cv2 as cv


def count_ventricular_area(img_array):
    """
    Oblicza powierzchnię przekroju komory
    :param img_array: obraz binarny komory
    :type img_array: array
    :return: powierzchnia komory
    :rtype: float
    """
    # Binarna maska komory
    ven_mask = cv.inRange(img_array, 1, 1)
    _, thresh_ven = cv.threshold(ven_mask, 2, 1, cv.THRESH_BINARY)
    contours_ven, _ = cv.findContours(thresh_ven, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ven_area_cnt = cv.contourArea(contours_ven[0])

    ven_area_cm = ven_area_cnt * 0.0308 * 0.0154

    return round(ven_area_cm, 2)


def count_wall_area(img_array):
    """
       Oblicza powierzchnię przekroju ściany
       :param img_array: obraz binarny ściany
       :type img_array: array
       :return: powierzchnia ściany
       :rtype: float
       """
    # Binarna maska ściany
    wall_mask = cv.inRange(img_array, 2, 2)
    _, thresh_wall = cv.threshold(wall_mask, 2, 1, cv.THRESH_BINARY)
    contours_wall, _ = cv.findContours(thresh_wall, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    wall_area = cv.contourArea(contours_wall[0])
    wall_area_cm = wall_area * 0.0308 * 0.0154

    return round(wall_area_cm, 2)
