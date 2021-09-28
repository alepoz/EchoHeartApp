def count_ventricular_mass(whole_volume, ven_volume):
    """
    Oblicza masę komory
    :param whole_volume: objętość komory ze ścianą
    :type whole_volume: float
    :param ven_volume: objętość samej komory
    :type ven_volume: float
    :return: mass: masa komory
    :rtype: float
    """
    mass = 1.05 * (whole_volume - ven_volume)
    return round(mass, 2)
