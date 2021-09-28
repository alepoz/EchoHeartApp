import numpy as np
import cv2.cv2 as cv
from Lib.LVAnalysis.wall_line_intersection import wall_line_intersection
from Lib.LVAnalysis.wall_lines_length import wall_lines_length
from Lib.LVAnalysis.wall_thickness import count_wall_thickness
from Lib.LVAnalysis.wall_thickness_ortogonal import count_wall_thickness_ortogonal
from Lib.LVAnalysis.math_functions import *


def wall_segmentation(img_array, images_out, apex, mid_base, contour_epi, contour_mid, contour_endo, view, standard):
    """
    Funkcja wyznacza segmenty ściany wg długości linii ściany oraz parametry tych segmentów
    :param img_array: obraz (maska)
    :type img_array: numpy.array
    :param images_out: obrazy wyjściowe
    :type images_out: list
    :param apex: współrzędne punktu koniuszka
    :type apex: tuple
    :param mid_base: współrzędne punktu środka podstawy
    :type mid_base: tuple
    :param contour_epi: kontur linii epicardialnej
    :type contour_epi: list
    :param contour_mid: kontur linii środkowej
    :type contour_mid: list
    :param contour_endo: kontur linii endocardialnej
    :type contour_endo: list
    :param view: projekcja (2CH lub 4CH)
    :type view: str
    :param standard: standard segmentacji (16-segments lub 17-segments)
    :type standard: str
    :returns images_out: obrazy wyjściowe, segments: segmenty, wall_thickness: średnia grubość ściany
    :rtype: list, dict, float
    """
    # Binarna maska komory
    ven_mask = cv.inRange(img_array, 1, 1)
    _, thresh_ven = cv.threshold(ven_mask, 1, 1, cv.THRESH_BINARY)
    contours_ven, _ = cv.findContours(thresh_ven, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Wyznaczenie środka ciężkości komory
    for cnt in contours_ven:
        M = cv.moments(cnt)
        mid_x = int(M["m10"] / (M["m00"] + 1e-5))
        mid_y = int(M["m01"] / (M["m00"] + 1e-5))

    # Wektor(linia) dzieląca komorę na część prawą i lewą
    y = np.arange(0, img_array.shape[0])
    line_v = np.vectorize(line_vector)
    x = line_v(y, apex[1], mid_base[1], apex[0], mid_base[0]).astype(np.uint)
    xy = list(zip(x, y))

    # Usuwanie części linii wykraczających poza wymiary obrazu
    dividing_line = [el for el in xy if 0 <= el[0] <= img_array.shape[1]]
    dl_pt1 = dividing_line[0]
    dl_pt2 = dividing_line[-1]

    # Maska lewej strony obrazu
    mask_l = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_left = np.array(
        [[0, 0], [0, img_array.shape[0]], dl_pt2, mid_base, [mid_x, mid_y], apex, dl_pt1])
    cv.fillPoly(mask_l, np.int32([pts_left]), (255))

    # Maska prawej strony obrazu
    mask_r = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_right = np.array(
        [dl_pt1, apex, [mid_x, mid_y], mid_base, dl_pt2, [img_array.shape[1], img_array.shape[0]],
         [img_array.shape[1], 0]])
    cv.fillPoly(mask_r, np.int32([pts_right]), (255))

    # Binarna maska ściany
    wall_mask = cv.inRange(img_array, 2, 2)
    _, thresh_wall = cv.threshold(wall_mask, 2, 1, cv.THRESH_BINARY)
    contours_wall, _ = cv.findContours(thresh_wall, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Podział ściany na stronę lewą i prawą
    img_left = cv.bitwise_and(thresh_wall, thresh_wall, mask=mask_l)
    img_right = cv.bitwise_and(thresh_wall, thresh_wall, mask=mask_r)

    # Detekcja punktu koniuszka ściany, komory w liniach epi i endo oraz ich przecięcie z linią środkową
    wall_apex = wall_line_intersection(contour_epi, dl_pt1, dl_pt2, step=5)
    apex = wall_line_intersection(contour_endo, dl_pt1, dl_pt2, step=5)

    # Podział linii endocardialnej i epicardialnej na część lewą oraz prawą wględem koniuszka
    left_endo_cnt = contour_endo[0:apex['idx']]
    left_endo_cnt.reverse()
    right_endo_cnt = contour_endo[apex['idx']::]
    left_epi_cnt = contour_epi[0:wall_apex['idx']]
    left_epi_cnt.reverse()
    right_epi_cnt = contour_epi[wall_apex['idx']::]

    # Obliczanie długości każdej z linii enocardialnej i epicardialej po stronie lewej i prawej
    left_endo_len = wall_lines_length(left_endo_cnt, 0, len(left_endo_cnt) - 1)
    right_endo_len = wall_lines_length(right_endo_cnt, 0, len(right_endo_cnt) - 1)
    left_epi_len = wall_lines_length(left_epi_cnt, 0, len(left_epi_cnt) - 1)
    right_epi_len = wall_lines_length(right_epi_cnt, 0, len(right_epi_cnt) - 1)

    # Podział linii ściany po prawej i lewej stronie na 3 części (16 segmentów) lub 4 (17 segmentów)

    # Lewa strona endocardial
    left_endo_points = []
    if standard == '17-segments':
        step = round(left_endo_len / 7, 2)
        apex_pt_l_endo = None
        for j in range(0, len(left_endo_cnt)):
            length = wall_lines_length(left_endo_cnt, 0, j, 5)
            if length > step and not apex_pt_l_endo:
                apex_pt_l_endo = {'idx': j - 1, 'pt': tuple(left_endo_cnt[j - 1])}
            if length > 3 * step and len(left_endo_points) == 0:
                left_endo_points.append(tuple(left_endo_cnt[j - 1]))
                # cv.circle(out, tuple(left_endo_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                apical_pt_l_endo = {'idx': j - 1, 'pt': tuple(left_endo_cnt[j - 1])}
            if length > 5 * step and len(left_endo_points) == 1:
                left_endo_points.append(tuple(left_endo_cnt[j - 1]))
                # cv.circle(out, tuple(left_endo_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                mid_pt_l_endo = {'idx': j - 1, 'pt': tuple(left_endo_cnt[j - 1])}
                break

    if standard == '16-segments':
        step = round(left_endo_len / 3, 2)
        for j in range(0, len(left_endo_cnt)):
            length = wall_lines_length(left_endo_cnt, 0, j, 5)
            if length > step and len(left_endo_points) == 0:
                left_endo_points.append(tuple(left_endo_cnt[j - 1]))
                apical_pt_l_endo = {'idx': j - 1, 'pt': tuple(left_endo_cnt[j - 1])}
            if length > 2 * step and len(left_endo_points) == 1:
                left_endo_points.append(tuple(left_endo_cnt[j - 1]))
                mid_pt_l_endo = {'idx': j - 1, 'pt': tuple(left_endo_cnt[j - 1])}
                break

    # Prawa strona endocardial
    right_endo_points = []
    if standard == '17-segments':
        step = round(right_endo_len / 7, 2)
        apex_pt_r_endo = None
        for j in range(0, len(right_endo_cnt)):
            length = wall_lines_length(right_endo_cnt, 0, j, 5)
            if length > step and not apex_pt_r_endo:
                # cv.circle(out, tuple(right_endo_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                apex_pt_r_endo = {'idx': j - 1, 'pt': tuple(right_endo_cnt[j - 1])}
            if length > 3 * step and len(right_endo_points) == 0:
                # cv.circle(out, tuple(right_endo_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                right_endo_points.append(tuple(right_endo_cnt[j - 1]))
                apical_pt_r_endo = {'idx': j - 1, 'pt': tuple(right_endo_cnt[j - 1])}
            if length > 5 * step and len(right_endo_points) == 1:
                # cv.circle(out, tuple(right_endo_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                right_endo_points.append(tuple(right_endo_cnt[j - 1]))
                mid_pt_r_endo = {'idx': j - 1, 'pt': tuple(right_endo_cnt[j - 1])}
                break

    if standard == '16-segments':
        step = round(right_endo_len / 3, 2)
        for j in range(0, len(right_endo_cnt)):
            length = wall_lines_length(right_endo_cnt, 0, j, 5)
            if length > step and len(right_endo_points) == 0:
                right_endo_points.append(tuple(right_endo_cnt[j - 1]))
                apical_pt_r_endo = {'idx': j - 1, 'pt': tuple(right_endo_cnt[j - 1])}
            if length > 2 * step and len(right_endo_points) == 1:
                right_endo_points.append(tuple(right_endo_cnt[j - 1]))
                mid_pt_r_endo = {'idx': j - 1, 'pt': tuple(right_endo_cnt[j - 1])}
                break

    # Lewa strona epicardial
    left_epi_points = []
    if standard == '17-segments':
        step = round(left_epi_len / 7, 2)
        apex_pt_l_epi = None
        for j in range(0, len(left_epi_cnt)):
            length = wall_lines_length(left_epi_cnt, 0, j, 5)
            if length > step and not apex_pt_l_epi:
                # cv.circle(out, tuple(left_epi_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                apex_pt_l_epi = {'idx': j - 1, 'pt': tuple(left_epi_cnt[j - 1])}
            if length > 3 * step and len(left_epi_points) == 0:
                # cv.circle(out, tuple(left_epi_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                left_epi_points.append(tuple(left_epi_cnt[j - 1]))
                apical_pt_l_epi = {'idx': j - 1, 'pt': tuple(left_epi_cnt[j - 1])}
            if length > 5 * step and len(left_epi_points) == 1:
                # cv.circle(out, tuple(left_epi_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                left_epi_points.append(tuple(left_epi_cnt[j - 1]))
                mid_pt_l_epi = {'idx': j - 1, 'pt': tuple(left_epi_cnt[j - 1])}
                break

    if standard == '16-segments':
        step = round(left_epi_len / 3, 2)
        for j in range(0, len(left_epi_cnt)):
            length = wall_lines_length(left_epi_cnt, 0, j, 5)
            if length > step and len(left_epi_points) == 0:
                left_epi_points.append(tuple(left_epi_cnt[j - 1]))
                apical_pt_l_epi = {'idx': j - 1, 'pt': tuple(left_epi_cnt[j - 1])}
            if length > 2 * step and len(left_epi_points) == 1:
                left_epi_points.append(tuple(left_epi_cnt[j - 1]))
                mid_pt_l_epi = {'idx': j - 1, 'pt': tuple(left_epi_cnt[j - 1])}
                break

    # Prawa strona epicardial
    right_epi_points = []
    if standard == '17-segments':
        step = round(right_epi_len / 7, 2)
        apex_pt_r_epi = None
        for j in range(0, len(right_epi_cnt)):
            length = wall_lines_length(right_epi_cnt, 0, j, 5)
            if length > step and not apex_pt_r_epi:
                # cv.circle(out, tuple(right_epi_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                apex_pt_r_epi = {'idx': j - 1, 'pt': tuple(right_epi_cnt[j - 1])}
            if length > 3 * step and len(right_epi_points) == 0:
                # cv.circle(out, tuple(right_epi_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                right_epi_points.append(tuple(right_epi_cnt[j - 1]))
                apical_pt_r_epi = {'idx': j - 1, 'pt': tuple(right_epi_cnt[j - 1])}
            if length > 5 * step and len(right_epi_points) == 1:
                # cv.circle(out, tuple(right_epi_cnt[j - 1]), radius=1, color=(0, 255, 0), thickness=-1)
                right_epi_points.append(tuple(right_epi_cnt[j - 1]))
                mid_pt_r_epi = {'idx': j - 1, 'pt': tuple(right_epi_cnt[j - 1])}
                break

    if standard == '16-segments':
        step = round(right_epi_len / 3, 2)
        for j in range(0, len(right_epi_cnt)):
            length = wall_lines_length(right_epi_cnt, 0, j, 5)
            if length > step and len(right_epi_points) == 0:
                right_epi_points.append(tuple(right_epi_cnt[j - 1]))
                apical_pt_r_epi = {'idx': j - 1, 'pt': tuple(right_epi_cnt[j - 1])}
            if length > 2 * step and len(right_epi_points) == 1:
                right_epi_points.append(tuple(right_epi_cnt[j - 1]))
                mid_pt_r_epi = {'idx': j - 1, 'pt': tuple(right_epi_cnt[j - 1])}
                break

    # Łączenie punktów epi i endo po strone lewej oraz prawej
    left_side = list(zip(left_epi_points, left_endo_points))
    right_side = list(zip(right_epi_points, right_endo_points))

    # Segment Apex (jeżeli podział na 17 segmentów)
    if standard == '17-segments':
        # Segment apex - część po lewej stronie ściany
        if apex_pt_l_epi['pt'][0] == apex_pt_l_endo['pt'][0]:
            y = np.arange(0, img_array.shape[0])
            x = line_v(y, apex_pt_l_epi['pt'][1],
                       apex_pt_l_endo['pt'][1], apex_pt_l_epi['pt'][0], apex_pt_l_endo['pt'][0]).astype(np.int64)
        else:
            x = np.arange(0, img_array.shape[1])
            y = line_v(x, apex_pt_l_epi['pt'][0], apex_pt_l_endo['pt'][0], apex_pt_l_epi['pt'][1],
                       apex_pt_l_endo['pt'][1]).astype(np.int64)
        xy = list(zip(x, y))
        nxy = [el for el in xy if
               0 < el[1] < img_array.shape[0]]  # Usuwanie części linii wykraczających poza wymiary obrazu
        apex_l_pt1 = nxy[0]
        apex_l_pt2 = nxy[-1]

        # Segment apex - część po prawej stronie ściany
        if apex_pt_r_epi['pt'][0] == apex_pt_r_endo['pt'][0]:
            y = np.arange(0, img_array.shape[0])
            x = line_v(y, apex_pt_r_epi['pt'][1],
                       apex_pt_r_endo['pt'][1], apex_pt_r_epi['pt'][0], apex_pt_r_endo['pt'][0]).astype(np.int64)
        else:
            x = np.arange(0, img_array.shape[1])
            y = line_v(x, apex_pt_r_epi['pt'][0], apex_pt_r_endo['pt'][0], apex_pt_r_epi['pt'][1],
                       apex_pt_r_endo['pt'][1]).astype(np.int64)
        xy = list(zip(x, y))
        nxy = [el for el in xy if
               0 < el[1] < img_array.shape[0]]  # Usuwanie części linii wykraczających poza wymiary obrazu

        if apex_pt_r_epi['pt'][0] > apex_pt_r_endo['pt'][0]:
            apex_r_pt1 = nxy[0]
            apex_r_pt2 = nxy[-1]
        else:
            apex_r_pt1 = nxy[-1]
            apex_r_pt2 = nxy[0]

        # Maska segmentu apex
        mask_apex = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        pts_apex = np.array([[0, 0], [img_array.shape[1], 0], list(apex_pt_r_epi['pt']), list(apex_pt_r_endo['pt']),
                             list(apex_pt_l_endo['pt']), list(apex_pt_l_epi['pt'])])
        cv.fillPoly(mask_apex, np.int32([pts_apex]), (255))
        img_apex = cv.bitwise_and(thresh_wall, thresh_wall, mask=mask_apex)
        apex_cnt_epi = left_epi_cnt[apex_pt_l_epi['idx']:0:-1] + right_epi_cnt[0:apex_pt_r_epi['idx']]
        apex_cnt_endo = left_endo_cnt[apex_pt_l_endo['idx']:0:-1] + right_endo_cnt[0:apex_pt_r_endo['idx']]

    # Segment Apical lewy
    x = np.arange(0, img_array.shape[1])
    y = line_v(x, left_side[0][0][0], left_side[0][1][0], left_side[0][0][1], left_side[0][1][1]).astype(np.int64)
    xy = list(zip(x, y))
    nxy = [el for el in xy if
           0 < el[1] < img_array.shape[0]]  # Usuwanie części linii wykraczających poza wymiary obrazu
    apical_l_pt1 = nxy[0]
    apical_l_pt2 = nxy[-1]

    # Segment Apical prawy
    x = np.arange(0, img_array.shape[1])
    y = line_v(x, right_side[0][1][0], right_side[0][0][0], right_side[0][1][1], right_side[0][0][1]).astype(np.int64)
    xy = list(zip(x, y))
    nxy = [el for el in xy if
           0 < el[1] < img_array.shape[0]]  # Usuwanie części linii wykraczających poza wymiary obrazu
    if right_side[0][0][0] > right_side[0][1][0]:
        apical_r_pt1 = nxy[0]
        apical_r_pt2 = nxy[-1]
    else:
        apical_r_pt1 = nxy[-1]
        apical_r_pt2 = nxy[0]
    nxy.clear()

    # Segment Mid lewy
    x = np.arange(0, img_array.shape[1])
    y = line_v(x, left_side[1][0][0], left_side[1][1][0], left_side[1][0][1], left_side[1][1][1]).astype(np.int64)
    xy = list(zip(x, y))
    nxy = [x for x in xy if 0 < x[1] < img_array.shape[0]]
    mid_l_pt1 = nxy[0]
    mid_l_pt2 = nxy[-1]

    # Segment Mid prawy
    y = line_v(x, right_side[1][1][0], right_side[1][0][0], right_side[1][1][1], right_side[1][0][1]).astype(np.int64)
    xy = list(zip(x, y))
    nxy = [el for el in xy if 0 < el[1] < img_array.shape[0]]
    if right_side[1][0][0] > right_side[1][1][0]:
        mid_r_pt1 = nxy[0]
        mid_r_pt2 = nxy[-1]
    else:
        mid_r_pt1 = nxy[-1]
        mid_r_pt2 = nxy[0]

        # Zastosowanie maski w celu otrzymania segmentu apical_l
    mask_apical_l = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    if standard == '17-segments':
        pts_apical_l = np.array([apex_l_pt1, apex_l_pt2, apical_l_pt2, apical_l_pt1])
    else:
        pts_apical_l = np.array([[0, 0], [img_array.shape[1], 0], apical_l_pt2, apical_l_pt1])
    cv.fillPoly(mask_apical_l, np.int32([pts_apical_l]), (255))
    img_apical_l = cv.bitwise_and(img_left, img_left, mask=mask_apical_l)

    # Zastosowanie maski w celu otrzymania segmentu prawy apical (apical_r)
    mask_apical_r = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    if standard == '17-segments':
        pts_apical_r = np.array([apex_r_pt1, apex_r_pt2, apical_r_pt2, apical_r_pt1])
    else:
        pts_apical_r = np.array([[0, 0], apical_r_pt1, apical_r_pt2, [img_array.shape[1], 0]])
    cv.fillPoly(mask_apical_r, np.int32([pts_apical_r]), (255))
    img_apical_r = cv.bitwise_and(img_right, img_right, mask=mask_apical_r)

    # Zastosowanie maski w celu otrzymania segmentu lewego mid_l
    mask_mid_l = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_mid_l = np.array([apical_l_pt2, mid_l_pt2, mid_l_pt1, apical_l_pt1])
    cv.fillPoly(mask_mid_l, np.int32([pts_mid_l]), (255))
    img_mid_l = cv.bitwise_and(img_left, img_left, mask=mask_mid_l)

    # Zastosowanie maski w celu otrzymania segmentu prawego mid (mid_r)
    mask_mid_r = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_mid_r = np.array([apical_r_pt1, apical_r_pt2, mid_r_pt2, mid_r_pt1])
    cv.fillPoly(mask_mid_r, np.int32([pts_mid_r]), (255))
    img_mid_r = cv.bitwise_and(img_right, img_right, mask=mask_mid_r)

    # Zastosowanie maski w celu otrzymania segmentu lewego basal_l
    mask_basal_l = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_basal_l = np.array([mid_l_pt1, [0, img_array.shape[0]], [img_array.shape[1], img_array.shape[0]], mid_l_pt2])
    cv.fillPoly(mask_basal_l, np.int32([pts_basal_l]), (255))
    img_basal_l = cv.bitwise_and(img_left, img_left, mask=mask_basal_l)

    # Zastosowanie maski w celu otrzymania segmentu prawego basal (basal_r)
    mask_basal_r = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_basal_r = np.array([mid_r_pt1, [0, img_array.shape[0]], [img_array.shape[1], img_array.shape[0]], mid_r_pt2])
    cv.fillPoly(mask_basal_r, np.int32([pts_basal_r]), (255))
    img_basal_r = cv.bitwise_and(img_right, img_right, mask=mask_basal_r)

    # Punkty przecięcia lini środkowej z granicami poszczególnych segmentów
    mid_pts_l = []
    mid_pts_r = []
    if standard == '17-segments':
        mid_pts_l.append(wall_line_intersection(contour_mid, apex_pt_l_epi['pt'], apex_pt_l_endo['pt'], 3))
        mid_pts_r.append(wall_line_intersection(contour_mid, apex_pt_r_epi['pt'], apex_pt_r_endo['pt'], 3))
    if standard == '16-segments':
        mid_apex = wall_line_intersection(contour_mid, wall_apex['pt'], apex['pt'], 3)
        mid_pts_l.append(mid_apex)
        mid_pts_r.append(mid_apex)

    for j in range(0, len(right_side)):
        mid_pts_l.append(wall_line_intersection(contour_mid, left_side[j][0], left_side[j][1], 3))
        mid_pts_r.append(wall_line_intersection(contour_mid, right_side[j][0], right_side[j][1], 3))

    # Utworzenie słowników przechowywujących dane poszczególnych segmentów
    apex_segment = {}
    apical_right_segment = {}
    apical_left_segment = {}
    mid_right_segment = {}
    mid_left_segment = {}
    basal_right_segment = {}
    basal_left_segment = {}

    # Obliczanie długości linii w segmencie prawym apical
    if standard == '17-segments':
        apex_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, 0, apex_pt_r_epi['idx']) + \
                                                       wall_lines_length(left_epi_cnt, 0, apex_pt_l_epi['idx']), 2)
        apex_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, 0, mid_pts_l[0]['idx'],
                                                                         mid_pts_r[0]['idx']), 2)
        apex_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_endo_cnt, 0, apex_pt_r_endo['idx']) + \
                                                        wall_lines_length(left_endo_cnt, 0, apex_pt_l_endo['idx']), 2)

        apical_left_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, apex_pt_l_epi['idx'],
                                                                                apical_pt_l_epi['idx']), 2)
        apical_left_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, mid_pts_l[1]['idx'],
                                                                                mid_pts_l[0]['idx']), 2)
        apical_left_segment['Endocardial line length [cm]'] = round(wall_lines_length(left_endo_cnt, apex_pt_l_endo['idx'],
                                                                                 apical_pt_l_endo['idx']), 2)

        apical_right_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, apex_pt_r_endo['idx'],
                                                                                 apical_pt_r_endo['idx']), 2)
        apical_right_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, mid_pts_r[0]['idx'],
                                                                                 mid_pts_r[1]['idx']), 2)
        apical_right_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, apex_pt_r_epi['idx'],
                                                                                  apical_pt_r_epi['idx']), 2)

    if standard == '16-segments':
        apical_left_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, 0,
                                                                                apical_pt_l_epi['idx']), 2)
        apical_left_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, mid_pts_l[1]['idx'],
                                                                                mid_pts_l[0]['idx']), 2)
        apical_left_segment['Endocardial line length [cm]'] = round(wall_lines_length(left_endo_cnt, 0,
                                                                                 apical_pt_l_endo['idx']), 2)
        apical_right_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, 0,
                                                                                 apical_pt_r_epi['idx']), 2)
        apical_right_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, mid_pts_r[0]['idx'],
                                                                                 mid_pts_r[1]['idx']), 2)
        apical_right_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_endo_cnt, 0,
                                                                                  apical_pt_r_endo['idx']), 2)

    # Obliczanie długości linii w segmentach mid

    mid_left_segment['Endocardial line length [cm]'] = round(wall_lines_length(left_endo_cnt, apical_pt_l_endo['idx'],
                                                                          mid_pt_l_endo['idx']), 2)
    mid_left_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, mid_pts_l[2]['idx'],
                                                                         mid_pts_l[1]['idx']), 2)
    mid_left_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, apical_pt_l_epi['idx'],
                                                                         mid_pt_l_epi['idx']), 2)

    mid_right_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_endo_cnt, apical_pt_r_endo['idx'],
                                                                           mid_pt_r_endo['idx']), 2)
    mid_right_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, mid_pts_r[1]['idx'],
                                                                          mid_pts_r[2]['idx']), 2)
    mid_right_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, apical_pt_r_epi['idx'],
                                                                          mid_pt_r_epi['idx']), 2)
    # Obliczanie długości linii w segmentach basal

    basal_left_segment['Endocardial line length [cm]'] = round(wall_lines_length(left_endo_cnt, mid_pt_l_endo['idx'],
                                                                            len(left_endo_cnt) - 1), 2)
    basal_left_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, 0, mid_pts_l[2]['idx']), 2)
    basal_left_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, mid_pt_l_epi['idx'],
                                                                           len(left_epi_cnt) - 1), 2)

    basal_right_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_endo_cnt, mid_pt_r_endo['idx'],
                                                                             len(right_endo_cnt) - 1), 2)
    basal_right_segment['Myocardial line length [cm]'] = round(wall_lines_length(contour_mid, mid_pts_r[2]['idx'],
                                                                            len(contour_mid) - 1), 2)
    basal_right_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, mid_pt_r_epi['idx'],
                                                                            len(right_epi_cnt) - 1), 2)

    # Średnia grubość segmentów
    if standard == '17-segments':
        apex_segment['Thickness [cm]'] = count_wall_thickness(apex_cnt_endo, apex_cnt_epi)

        apical_left_segment['Thickness [cm]'] = count_wall_thickness(left_endo_cnt[
                                                                apex_pt_l_endo['idx']:apical_pt_l_endo['idx'] + 1],
                                                                left_epi_cnt[
                                                                apex_pt_l_epi['idx']:apical_pt_l_epi['idx'] + 1])
        apical_right_segment['Thickness [cm]'] = count_wall_thickness(right_endo_cnt[
                                                                 apex_pt_r_endo['idx']:apical_pt_r_endo['idx'] + 1],
                                                                 right_epi_cnt[
                                                                 apex_pt_r_epi['idx']:apical_pt_r_epi['idx'] + 1])
    if standard == '16-segments':
        apical_left_segment['Thickness [cm]'] = count_wall_thickness(left_endo_cnt[0:apical_pt_l_endo['idx'] + 1],
                                                                left_epi_cnt[0:apical_pt_l_epi['idx'] + 1])
        apical_right_segment['Thickness [cm]'] = count_wall_thickness(right_endo_cnt[0:apical_pt_r_endo['idx'] + 1],
                                                                 right_epi_cnt[0:apical_pt_r_epi['idx'] + 1])

    mid_left_segment['Thickness [cm]'] = count_wall_thickness(left_endo_cnt[
                                                         apical_pt_l_endo['idx']:mid_pt_l_endo['idx'] + 1],
                                                         left_epi_cnt[apical_pt_l_epi['idx']: mid_pt_l_epi['idx'] + 1])
    mid_right_segment['Thickness [cm]'] = count_wall_thickness(right_endo_cnt[
                                                          apical_pt_r_endo['idx']:mid_pt_r_endo['idx'] + 1],
                                                          right_epi_cnt[
                                                          apical_pt_r_epi['idx']: mid_pt_r_epi['idx'] + 1])
    basal_left_segment['Thickness [cm]'] = count_wall_thickness(left_endo_cnt[mid_pt_l_endo['idx']:len(left_endo_cnt)],
                                                           left_epi_cnt[mid_pt_l_epi['idx']:len(left_epi_cnt)])
    basal_right_segment['Thickness [cm]'] = count_wall_thickness(right_endo_cnt[mid_pt_r_endo['idx']:len(right_endo_cnt)],
                                                            right_epi_cnt[mid_pt_r_epi['idx']: len(right_epi_cnt)])

    # Odznaczenie segmentów na obrazie
    for img_out in images_out:
        img_out[img_apical_l == 1] = [204, 255, 153]
        img_out[img_mid_l == 1] = [153, 255, 51]
        img_out[img_basal_l == 1] = [102, 204, 0]
        img_out[img_apical_r == 1] = [153, 255, 255]
        img_out[img_mid_r == 1] = [51, 255, 255]
        img_out[img_basal_r == 1] = [0, 204, 204]
        # Zaznaczenie granic segmentów na obrazie
        cv.line(img_out, left_side[0][0], left_side[0][1], (150, 150, 150), 1, cv.LINE_AA)
        cv.line(img_out, left_side[1][0], left_side[1][1], (150, 150, 150), 1, cv.LINE_AA)
        cv.line(img_out, right_side[0][0], right_side[0][1], (150, 150, 150), 1, cv.LINE_AA)
        cv.line(img_out, right_side[1][0], right_side[1][1], (150, 150, 150), 1, cv.LINE_AA)

    # Powierzchnia segmentów
    if standard == '17-segments':
        contours_apex, _ = cv.findContours(thresh_wall, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        apex_segment['Area [cm^2]'] = round(cv.contourArea(contours_apex[0]) * 0.0308 * 0.0154, 2)
        for img_out in images_out:
            img_out[img_apex == 1] = [255, 255, 204]
            cv.line(img_out, apex_pt_l_epi['pt'], apex_pt_l_endo['pt'], (150, 150, 150), 1, cv.LINE_AA)
            cv.line(img_out, apex_pt_r_epi['pt'], apex_pt_r_endo['pt'], (150, 150, 150), 1, cv.LINE_AA)
    if standard == '16-segments':
        for img_out in images_out:
            cv.line(img_out, tuple(wall_apex['pt']), tuple(apex['pt']), (150, 150, 150), 1, cv.LINE_AA)

    # Kontury segmentów
    contours_basal_l, _ = cv.findContours(img_basal_l, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_mid_l, _ = cv.findContours(img_mid_l, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_apical_l, _ = cv.findContours(img_apical_l, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_apical_r, _ = cv.findContours(img_apical_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_mid_r, _ = cv.findContours(img_mid_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_basal_r, _ = cv.findContours(img_basal_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Obliczenie i zapis powierzchni poszczególnych segmentów
    apical_right_segment['Area [cm^2]'] = round(cv.contourArea(contours_apical_r[0]) * 0.0308 * 0.0154, 2)
    apical_left_segment['Area [cm^2]'] = round(cv.contourArea(contours_apical_l[0]) * 0.0308 * 0.0154, 2)
    mid_right_segment['Area [cm^2]'] = round(cv.contourArea(contours_mid_r[0]) * 0.0308 * 0.0154, 2)
    mid_left_segment['Area [cm^2]'] = round(cv.contourArea(contours_mid_l[0]) * 0.0308 * 0.0154, 2)
    basal_right_segment['Area [cm^2]'] = round(cv.contourArea(contours_basal_r[0]) * 0.0308 * 0.0154, 2)
    basal_left_segment['Area [cm^2]'] = round(cv.contourArea(contours_basal_l[0]) * 0.0308 * 0.0154, 2)

    segments = {'Method': "Section by line's length", 'Standard': standard}
    if standard == '17-segments':
        segments['Apex'] = apex_segment

    if view == '4CH':
        # Lewa strona
        segments['Apical septal'] = apical_left_segment
        segments['Mid inferoseptal'] = mid_left_segment
        segments['Basal inferoseptal'] = basal_left_segment
        # Prawa strona
        segments['Apical lateral'] = apical_right_segment
        segments['Mid anterolateral'] = mid_right_segment
        segments['Basal anterolateral'] = basal_right_segment
    if view == '2CH':
        # Lewa strona
        segments['Apical inferior'] = apical_left_segment
        segments['Mid inferior'] = mid_left_segment
        segments['Basal inferior'] = basal_left_segment
        # Prawa strona
        segments['Apical anterior'] = apical_right_segment
        segments['Mid anterior'] = mid_right_segment
        segments['Basal anterior'] = basal_right_segment

    wall_thickness_l = count_wall_thickness_ortogonal(img_left, left_endo_cnt, left_epi_cnt)
    wall_thickness_r = count_wall_thickness_ortogonal(img_right, right_endo_cnt, right_epi_cnt)
    wall_thickness = round((wall_thickness_l + wall_thickness_r) / 2, 2)

    return images_out, segments, wall_thickness
