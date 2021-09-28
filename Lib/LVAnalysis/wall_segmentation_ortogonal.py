import numpy as np
import cv2.cv2 as cv
from Lib.LVAnalysis.wall_line_intersection import wall_line_intersection
from Lib.LVAnalysis.wall_lines_length import wall_lines_length
from Lib.LVAnalysis.wall_thickness import count_wall_thickness
from Lib.LVAnalysis.wall_thickness_ortogonal import count_wall_thickness_ortogonal
from Lib.LVAnalysis.math_functions import *


def wall_segmentation_ortogonal(img_array, images_out, apex, mid_base, contour_epi, contour_mid, contour_endo, view,
                                standard):
    """
       Funkcja wyznacza segmenty ściany wg prostych ortogonalnych do osi długiej komory oraz parametry tych segmentów
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
    # Binarna maska ściany
    wall_mask = cv.inRange(img_array, 2, 2)
    _, thresh_wall = cv.threshold(wall_mask, 2, 1, cv.THRESH_BINARY)
    contours_wall, _ = cv.findContours(thresh_wall, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Binarna maska komory
    ven_mask = cv.inRange(img_array, 1, 1)
    _, thresh_ven = cv.threshold(ven_mask, 1, 1, cv.THRESH_BINARY)
    contours_ven, _ = cv.findContours(thresh_ven, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Wyznaczenie środka ciężkości komory
    for cnt in contours_ven:
        M = cv.moments(cnt)
        mid_x = int(M["m10"] / (M["m00"] + 1e-5))
        mid_y = int(M["m01"] / (M["m00"] + 1e-5))

    # Wektor(linia) dzieląca obraz wdłuż osi głównej na część lewą i prawą
    y = np.arange(0, img_array.shape[0])
    line_v = np.vectorize(line_vector)
    x = line_v(y, apex[1], mid_base[1], apex[0], mid_base[0]).astype(np.uint)
    xy = list(zip(x, y))

    # Usuwanie części linii wykraczających poza wymiary obrazu
    dividing_line = [el for el in xy if 0 <= el[0] <= img_array.shape[1]]
    dl_pt1 = dividing_line[0]
    dl_pt2 = dividing_line[-1]

    # Wycinannie lewej strony ściany
    mask_l = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_left = np.array(
        [[0, 0], [0, img_array.shape[0]], dl_pt2, mid_base, [mid_x, mid_y], apex, dl_pt1])
    cv.fillPoly(mask_l, np.int32([pts_left]), (255))

    # Wycinannie prawej strony ściany
    mask_r = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_right = np.array(
        [dl_pt1, apex, [mid_x, mid_y], mid_base, dl_pt2, [img_array.shape[1], img_array.shape[0]],
         [img_array.shape[1], 0]])
    cv.fillPoly(mask_r, np.int32([pts_right]), (255))

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

    # Podział osi na 3 części wyznaczające segmenty: apical, mid oraz basal
    axis= []
    for t in np.arange(1 / 3, 0.9, 1 / 3):  # podział na 3 części
        Cx = int(apex['pt'][0] * (1 - t) + mid_base[0] * t)
        Cy = int(apex['pt'][1] * (1 - t) + mid_base[1] * t)
        axis.append((Cx, Cy))

    # Podział lini środkowej na prawą i lewą stronę
    mid_apex = wall_line_intersection(contour_mid, wall_apex['pt'], apex['pt'], 2)
    left_mid_cnt = contour_mid[0:mid_apex['idx']]
    left_mid_cnt.reverse()
    right_mid_cnt = contour_mid[mid_apex['idx']::]

    line_vec = np.vectorize(ortogonal_vector)
    segment_lines = []
    slope, _, _ = ortogonal(wall_apex['pt'], mid_base)
    left_side = []
    right_side = []
    mid_pts_l = []
    mid_pts_r = []

    if standard == '17-segments':
        sx = np.arange(0, img_array.shape[1])
        sy = line_vec(sx, slope, apex['pt'][0], apex['pt'][1]).astype(np.uint)

        sxy = list(zip(sx, sy))
        seg_line = [el for el in sxy if 0 < el[1] < img_array.shape[0]]
        apex_line_l = seg_line[0]
        apex_line_r = seg_line[-1]
        apex_r_epi_pt = wall_line_intersection(right_epi_cnt, seg_line[0], seg_line[-1], step=5)
        apex_l_epi_pt = wall_line_intersection(left_epi_cnt, seg_line[0], seg_line[-1], step=5)
        apex_r_mid_pt = wall_line_intersection(right_mid_cnt, seg_line[0], seg_line[-1], step=3)
        apex_l_mid_pt = wall_line_intersection(left_mid_cnt, seg_line[0], seg_line[-1], step=3)

    for j in range(0, len(axis)):
        sx = np.arange(0, img_array.shape[1])
        sy = line_vec(sx, slope, axis[j][0], axis[j][1]).astype(np.uint)
        sxy = list(zip(sx, sy))
        seg_line = [el for el in sxy if 0 < el[1] < img_array.shape[0]]

        segment_lines.append((seg_line[0], seg_line[-1]))
        r_epi_pt = wall_line_intersection(right_epi_cnt, seg_line[0], seg_line[-1], step=5)
        l_epi_pt = wall_line_intersection(left_epi_cnt, seg_line[0], seg_line[-1], step=5)

        r_endo_pt = wall_line_intersection(right_endo_cnt, seg_line[0], seg_line[-1], step=5)
        l_endo_pt = wall_line_intersection(left_endo_cnt, seg_line[0], seg_line[-1], step=5)

        mid_pts_r.append(wall_line_intersection(right_mid_cnt, seg_line[0], seg_line[-1], step=3))
        mid_pts_l.append(wall_line_intersection(left_mid_cnt, seg_line[0], seg_line[-1], step=3))

        left_side.append((l_epi_pt, l_endo_pt))
        right_side.append((r_epi_pt, r_endo_pt))

    # Metooda 17-segmentowa - dodatkowy segment apex
    if standard == '17-segments':
        mask_apex = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        pts_apex = np.array([[0, 0], [img_array.shape[1], 0], segment_lines[0][1], segment_lines[0][0]])
        cv.fillPoly(mask_apex, np.int32([pts_apex]), (255))
        img_apex = cv.bitwise_and(thresh_wall, thresh_wall, mask=mask_apex)

    # Zastosowanie maski w celu otrzymania segmentu lewego i prawego apical
    mask_apical = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    if standard == '16-segments':
        pts_apical = np.array([[0, 0], [img_array.shape[1], 0], segment_lines[0][1], segment_lines[0][0]])
    if standard == '17-segments':
        pts_apical = np.array([apex_line_l, apex_line_r, segment_lines[0][1], segment_lines[0][0]])
    cv.fillPoly(mask_apical, np.int32([pts_apical]), (255))
    img_apical_l = cv.bitwise_and(img_left, img_left, mask=mask_apical)
    img_apical_r = cv.bitwise_and(img_right, img_right, mask=mask_apical)

    # Zastosowanie maski w celu otrzymania segmentu lewego i prawego mid
    mask_mid = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_mid = np.array([segment_lines[0][0], segment_lines[0][1], segment_lines[1][1], segment_lines[1][0]])
    cv.fillPoly(mask_mid, np.int32([pts_mid]), (255))
    img_mid_l = cv.bitwise_and(img_left, img_left, mask=mask_mid)
    img_mid_r = cv.bitwise_and(img_right, img_right, mask=mask_mid)

    # Zastosowanie maski w celu otrzymania segmentu lewego i prawego basal
    mask_basal = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    pts_basal_l = np.array([segment_lines[1][0], segment_lines[1][1], [img_array.shape[1], img_array.shape[0]],
                            [0, img_array.shape[0]]])
    cv.fillPoly(mask_basal, np.int32([pts_basal_l]), (255))
    img_basal_l = cv.bitwise_and(img_left, img_left, mask=mask_basal)
    img_basal_r = cv.bitwise_and(img_right, img_right, mask=mask_basal)

    for img_out in images_out:
        if standard == '17-segments':
            img_out[img_apex == 1] = [255, 255, 204]
        img_out[img_basal_l == 1] = [102, 204, 0]
        img_out[img_mid_l == 1] = [153, 255, 51]
        img_out[img_apical_l == 1] = [204, 255, 153]
        img_out[img_apical_r == 1] = [153, 255, 255]
        img_out[img_mid_r == 1] = [51, 255, 255]
        img_out[img_basal_r == 1] = [0, 204, 204]

    apex_segment = {}
    apical_right_segment = {}
    apical_left_segment = {}
    mid_right_segment = {}
    mid_left_segment = {}
    basal_right_segment = {}
    basal_left_segment = {}

    # 17 segmentów - Obliczanie długości linii w segmentach apex i apical
    if standard == '17-segments':
        apex_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, 0,
                                                                         apex_l_epi_pt['idx'], 5) +
                                                       wall_lines_length(right_epi_cnt, 0,
                                                                         apex_r_epi_pt['idx'], 5), 2)
        apex_segment['Myocardial line length [cm]'] = round(wall_lines_length(left_mid_cnt, 0, apex_l_mid_pt['idx'], 2) +
                                                       wall_lines_length(right_mid_cnt, 0,
                                                                         apex_r_mid_pt['idx'], 2), 2)
        apex_segment['Endocardial line length [cm]'] = 0

        apical_left_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, apex_l_epi_pt['idx'],
                                                                                left_side[0][0]['idx'], 5), 2)
        apical_left_segment['Myocardial line length [cm]'] = round(wall_lines_length(left_mid_cnt, apex_l_mid_pt['idx'],
                                                                                mid_pts_l[0]['idx'], 2), 2)
        apical_left_segment['Endocardial line length [cm]'] = round(wall_lines_length(left_endo_cnt, 0,
                                                                                 left_side[0][1]['idx'], 5), 2)

        apical_right_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, apex_r_epi_pt['idx'],
                                                                                 right_side[0][0]['idx'], 5), 2)
        apical_right_segment['Myocardial line length [cm]'] = round(wall_lines_length(right_mid_cnt, apex_r_mid_pt['idx'],
                                                                                 mid_pts_r[0]['idx'], 2), 2)
        apical_right_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_endo_cnt, 0,
                                                                                  right_side[0][1]['idx'], 5), 2)
    # 16 segmentów - Obliczanie długości linii w segmentach apical
    if standard == '16-segments':
        apical_left_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, 0,
                                                                                left_side[0][0]['idx'], 5), 2)
        apical_left_segment['Myocardial line length [cm]'] = round(wall_lines_length(left_mid_cnt, 0,
                                                                                mid_pts_l[0]['idx'], 2), 2)
        apical_left_segment['Endocardial line length [cm]'] = round(wall_lines_length(left_endo_cnt, 0,
                                                                                 left_side[0][1]['idx'], 5), 2)

        apical_right_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, 0,
                                                                                 right_side[0][0]['idx'], 5), 2)
        apical_right_segment['Myocardial line length [cm]'] = round(wall_lines_length(right_mid_cnt, 0,
                                                                                 mid_pts_r[0]['idx'], 2), 2)
        apical_right_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_endo_cnt, 0,
                                                                                  right_side[0][1]['idx'], 5), 2)

    # Obliczanie długości linii w segmentach mid
    mid_left_segment['Endocardial line length [cm]'] = round(wall_lines_length(left_endo_cnt, left_side[0][1]['idx'],
                                                                          left_side[1][1]['idx'], 5), 2)
    mid_left_segment['Myocardial line length [cm]'] = round(wall_lines_length(left_mid_cnt, mid_pts_l[0]['idx'],
                                                                         mid_pts_l[1]['idx'], 2), 2)
    mid_left_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, left_side[0][0]['idx'],
                                                                         left_side[1][0]['idx'], 5), 2)

    mid_right_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_endo_cnt, right_side[0][1]['idx'],
                                                                           right_side[1][1]['idx'], 5), 2)
    mid_right_segment['Myocardial line length [cm]'] = round(wall_lines_length(right_mid_cnt, mid_pts_r[0]['idx'],
                                                                          mid_pts_r[1]['idx'], 2), 2)
    mid_right_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, right_side[0][0]['idx'],
                                                                          right_side[1][0]['idx'], 5), 2)

    # Obliczanie długości linii w segmenach basal
    basal_left_segment['Endocardial line length [cm]'] = round(wall_lines_length(left_endo_cnt, left_side[1][1]['idx'],
                                                                            len(left_endo_cnt) - 1, 5), 2)
    basal_left_segment['Myocardial line length [cm]'] = round(wall_lines_length(left_mid_cnt, mid_pts_l[1]['idx'],
                                                                           len(left_mid_cnt) - 1, 2), 2)
    basal_left_segment['Epicardial line length [cm]'] = round(wall_lines_length(left_epi_cnt, left_side[1][0]['idx'],
                                                                           len(left_epi_cnt) - 1, 5), 2)

    basal_right_segment['Endocardial line length [cm]'] = round(wall_lines_length(right_endo_cnt, right_side[1][1]['idx'],
                                                                             len(right_endo_cnt) - 1, 5), 2)
    basal_right_segment['Myocardial line length [cm]'] = round(wall_lines_length(right_mid_cnt, mid_pts_r[1]['idx'],
                                                                            len(right_mid_cnt) - 1, 2), 2)
    basal_right_segment['Epicardial line length [cm]'] = round(wall_lines_length(right_epi_cnt, right_side[1][0]['idx'],
                                                                            len(right_epi_cnt) - 1, 5), 2)

    # Średnia grubość segmentów

    if standard == '17-segments':
        apex_segment['Thickness [cm]'] = None
        apical_left_segment['Thickness [cm]'] = count_wall_thickness(left_endo_cnt[0:left_side[0][1]['idx'] + 1],
                                                                left_epi_cnt[
                                                                apex_l_epi_pt['idx']:left_side[0][0]['idx'] + 1])
        apical_right_segment['Thickness [cm]'] = count_wall_thickness(right_endo_cnt[0:right_side[0][1]['idx'] + 1],
                                                                 right_epi_cnt[
                                                                 apex_r_epi_pt['idx']:right_side[0][0]['idx'] + 1])
    if standard == '16-segments':
        apical_left_segment['Thickness [cm]'] = count_wall_thickness(left_endo_cnt[0:left_side[0][1]['idx'] + 1],
                                                                left_epi_cnt[0:left_side[0][0]['idx'] + 1])
        apical_right_segment['Thickness [cm]'] = count_wall_thickness(right_endo_cnt[0:right_side[0][1]['idx'] + 1],
                                                                 right_epi_cnt[0:right_side[0][0]['idx'] + 1])

    mid_left_segment['Thickness [cm]'] = count_wall_thickness(left_endo_cnt[
                                                         left_side[0][1]['idx']:left_side[1][1]['idx'] + 1],
                                                         left_epi_cnt[
                                                         left_side[0][0]['idx']:left_side[1][0]['idx'] + 1])
    mid_right_segment['Thickness [cm]'] = count_wall_thickness(right_endo_cnt[
                                                          right_side[0][1]['idx']:right_side[1][1]['idx'] + 1],
                                                          right_epi_cnt[
                                                          right_side[0][0]['idx']:right_side[1][0]['idx'] + 1])
    basal_left_segment['Thickness [cm]'] = count_wall_thickness(left_endo_cnt[left_side[1][1]['idx']:len(left_endo_cnt)],
                                                           left_epi_cnt[left_side[1][0]['idx']:len(left_epi_cnt)])
    basal_right_segment['Thickness [cm]'] = count_wall_thickness(right_endo_cnt[right_side[1][1]['idx']:len(right_endo_cnt)],
                                                            right_epi_cnt[right_side[1][0]['idx']: len(right_epi_cnt)])

    # Kontury segmentów
    contours_basal_l, _ = cv.findContours(img_basal_l, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_mid_l, _ = cv.findContours(img_mid_l, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_apical_l, _ = cv.findContours(img_apical_l, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_apical_r, _ = cv.findContours(img_apical_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_mid_r, _ = cv.findContours(img_mid_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_basal_r, _ = cv.findContours(img_basal_r, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Obliczenie i zapis powierzchni poszczególnych segmentów
    if standard == '17-segments':
        contours_apex, _ = cv.findContours(img_apex, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        apex_segment['Area [cm^2]'] = round(cv.contourArea(contours_apex[0]) * 0.0308 * 0.0154, 2)

    apical_right_segment['Area [cm^2]'] = round(cv.contourArea(contours_apical_r[0]) * 0.0308 * 0.0154, 2)
    apical_left_segment['Area [cm^2]'] = round(cv.contourArea(contours_apical_l[0]) * 0.0308 * 0.0154, 2)
    mid_right_segment['Area [cm^2]'] = round(cv.contourArea(contours_mid_r[0]) * 0.0308 * 0.0154, 2)
    mid_left_segment['Area [cm^2]'] = round(cv.contourArea(contours_mid_l[0]) * 0.0308 * 0.0154, 2)
    basal_right_segment['Area [cm^2]'] = round(cv.contourArea(contours_basal_r[0]) * 0.0308 * 0.0154, 2)
    basal_left_segment['Area [cm^2]'] = round(cv.contourArea(contours_basal_l[0]) * 0.0308 * 0.0154, 2)

    for img_out in images_out:
        if standard == '16-segments':
            cv.line(img_out, wall_apex['pt'], apex['pt'], (150, 150, 150), 1, cv.LINE_AA)
        if standard == '17-segments':
            cv.line(img_out, apex_l_epi_pt['pt'], apex_r_epi_pt['pt'], (150, 150, 150), 1, cv.LINE_AA)

        for i in range(0, len(left_side)):
            cv.line(img_out, tuple(left_side[i][0]['pt']), tuple(left_side[i][1]['pt']), (150, 150, 150), 1, cv.LINE_AA)
            cv.line(img_out, tuple(right_side[i][0]['pt']), tuple(right_side[i][1]['pt']), (150, 150, 150), 1,
                    cv.LINE_AA)

    segments = {'Method': "Section by ortagonal lines to axis", 'Standard': standard}
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

    # Olbiczanie średniej grubości ściany
    wall_thickness_l = count_wall_thickness_ortogonal(thresh_wall, left_endo_cnt, left_epi_cnt)
    wall_thickness_r = count_wall_thickness_ortogonal(thresh_wall, right_endo_cnt, right_epi_cnt)
    wall_thickness = round((wall_thickness_l + wall_thickness_r) / 2, 2)

    return images_out, segments, wall_thickness
