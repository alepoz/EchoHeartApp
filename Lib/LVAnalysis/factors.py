def count_gls(es_length, ed_length):
    """
    Oblicza współczynnik GLS
    :param es_length: długość linii ściany w fazie końcowo-skurczowej
    :type es_length: float
    :param ed_length: długość linii ściany w fazie końcowo-rozkurczowej
    :type ed_length: float
    :return: współczynnik gls
    :rtype: float
    """
    gls = round((es_length - ed_length) * 100 / ed_length, 2)
    return gls


def count_ejection_fraction(es_volume, ed_volume):
    """
    Oblicza frakcję wyrzutową
    :param es_volume: objętość komory w fazie końcowo-skurczowej
    :type es_volume: float
    :param ed_volume: objętość komory w fazie końcowo-rozkurczowej
    :type ed_volume: float
    :return: współczynnik fakcji wyrzutowej
    :rtype: float
    """
    ej = round((ed_volume - es_volume)*100/ed_volume, 2)
    return ej
