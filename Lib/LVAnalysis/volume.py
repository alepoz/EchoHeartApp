from Lib.LVAnalysis.simpson_method import simpson_disks_ventricular, simpson_disks_wall
import math


def count_volume(img_array2CH, img_array4CH, data2CH, data4CH, organ):
    """
    Funkcja oblicza objętość obiektu metodą Simpsona
    :param img_array2CH: obraz w projekcji 2CH
    :type img_array2CH: numpy.array
    :param img_array4CH: obraz w projekcji 4CH
    :type img_array4CH: numpy.array
    :param data2CH: parametry projekcji 2CH - położenie koniuszka i punktów podstaw
    :type data2CH: dict
    :param data4CH: parametry projekcji 4CH - położenie koniuszka i punktów podstaw
    :type data4CH: dict
    :param organ: narząd którego objętość jest liczona - 'ventricular' (komora) lub 'whole' (komora + ściana)
    :type organ: str
    :return: volume: objętość
    :rtype: float
    """
    if organ == 'ventricular':
        d_2CH, h_2CH = simpson_disks_ventricular(img_array2CH, data2CH)
        d_4CH, h_4CH = simpson_disks_ventricular(img_array4CH, data4CH)
    if organ == 'whole':
        d_2CH, h_2CH = simpson_disks_wall(img_array2CH, data2CH)
        d_4CH, h_4CH = simpson_disks_wall(img_array4CH, data4CH)
    # Uśrednianie wysokości
    avgh = []
    heights = list(zip(h_2CH, h_4CH))
    for h in heights:
        avgh.append((h[0] + h[1]) / 2)

    # Obliczanie objętości
    volume = 0
    for j in range(0, len(d_2CH)):
        volume = volume + math.pi * d_2CH[j] * d_4CH[j] * avgh[j] / 4

    return round(volume, 2)
