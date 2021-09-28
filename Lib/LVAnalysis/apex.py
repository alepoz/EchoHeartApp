import cv2.cv2 as cv
from Lib.LVAnalysis.math_functions import *


def detect_apex(img_array, images_out, mid_base):
    """
       Wyznacza położenie punktu koniuszka
       :param img_array: obraz maski
       :type img_array: array
       :param images_out: obrazy wyjściowe
       :type images_out: list
       :param mid_base: punkt położenia środka podstawy
       :type mid_base: tuple/list
       :return: punkt położenia koniuszka
       :rtype: tuple
       """
    ventricle_mask = cv.inRange(img_array, 1, 1)
    ret_ven, thresh_ven = cv.threshold(ventricle_mask, 2, 1, cv.THRESH_BINARY)
    contours_ven, hierarchy_ven = cv.findContours(thresh_ven, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours_ven:
        moments = cv.moments(cnt)
        cy = int(moments["m01"] / (moments["m00"] + 1e-5))

    max_dist = 0
    apex = None
    for y in range(cy, 0, -1):
        line = list(img_array[y])
        if apex and line.count(1) == 0:
            break
        for x in range(1, thresh_ven.shape[1]):
            if thresh_ven[y][x] == 1:
                dist = distance((x, y), mid_base)
                if dist > max_dist:
                    max_dist = dist
                    apex = (x, y)

    for img_out in images_out:
        cv.circle(img_out, apex, radius=3, color=(0, 255, 0), thickness=-1)

    return images_out, apex
