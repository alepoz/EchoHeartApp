from src.patient import Patient


class Model(object):

    def __init__(self):
        self.patient = None
        self.current_img_index = 0  # indeks aktualnie wyświetlanego pacjenta

    def add_patient(self, info):
        """
        Dodaje pacjenta
        :param info: informacje o pacjencie
        :type info: dict
        :return:
        """
        if self.patient:
            del self.patient
        self.patient = Patient(info)

    def get_patient(self):
        """
        Zwraca obiekt pacjent
        :return: patient: pacjent
        :rtype: Patient
        """
        return self.patient
