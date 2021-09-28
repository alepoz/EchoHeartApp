from src.sitkImage import SitkImage


class Patient(object):

    def __init__(self, personal_information):
        self.number = personal_information['PatientNumber']
        self.age = personal_information['Age']
        if personal_information['Sex'] == 'F':
            self.sex = 'Female'
        else:
            self.sex = 'Male'
        self.images = []

    # Dodaj pojedynczy obraz
    def add_image(self, image, data):
        """
        Funkcja dodająca obraz do listy obrazów pacjenta
        :param image: obraz
        :type image numpy.array
        :param data: dane obrazu
        :type data: dict
        :return: None
        """
        self.images.append(SitkImage(image, data))

    # Zwróć obraz wg nazwy
    def get_image_by_name(self, name):
        """
        Funkcja zwracająca obraz pacjenta wg podanej nazwy
        :param name: nazwa obrazu
        :type name: str
        :return: img: obraz
        :rtype: numpy.array
        """
        for img in self.images:
            if img.get_name() == name:
                return img

    # Zwróć obraz wg indeksu
    def get_image_by_idx(self, img_index):
        """
        Fukcja zwracająca obraz pacjenta wg podanego indeksu w liście
        :param img_index: indeks obrazu
        :type img_index: int
        :return: images[img_index]: obraz
        :rtype: numpy.array
        """
        return self.images[img_index]

    # Zwróć pierwszy obraz z listy
    def get_single_image(self):
        """
            Fukcja zwracająca pierwszy obraz pacjenta z listy
            :return: images[0]: obraz
            :rtype: numpy.array
        """
        return self.images[0]

    # Zwróć listę wszystkich obrazów
    def get_images(self):
        """ Fukcja zwracająca listę obrazów pacjenta
            :return: images: lista obrazów
            :rtype: list
        """
        return self.images

    # Zwróć obrazy wg projekcji
    def get_images_by_projection(self, projection):
        """
        Fukcja zwracająca listę obrazówwg projekcji
        :param projection: projekcja
        :type projection: str
        :return: images: lista obrazów
        :rtype: list
        """
        images = []
        for img in self.images:
            if img.get_projection() == projection:
                images.append(img)
        return images

    # Zwróć obrazy wg fazy
    def get_images_by_phase(self, phase):
        """ Fukcja zwracająca listę obrazówwg projekcji
            :param phase: faza
            :type phase: str
            :return: images: lista obrazów
            :rtype: list
        """
        images = []
        for img in self.images:
            if img.get_phase() == phase:
                images.append(img)
        return images

    # Zwróć obrazy wg typu
    def get_images_by_type(self, img_type):
        """
            Fukcja zwracająca listę obrazów wg typu
            :param img_type: typ
            :type img_type: str
            :return: images: lista obrazów
            :rtype: list
        """
        images = []
        for img in self.images:
            if img.get_img_type() == img_type:
                images.append(img)
        return images

    # Zwróć numer pacjenta
    def get_number(self):
        """
        Funkcja zwracająca numer pacjenta
        :return: number: numer
        :rtype: str
        """
        return self.number

    # Zwróć wiek pacjenta
    def get_age(self):
        """
        Funkcja zwracająca wiek pacjenta
        :return: age: wiek
        :rtype: str
        """
        return self.age

    # Zwróć płeć pacjenta
    def get_sex(self):
        """
       Funkcja zwracająca płeć pacjenta
       :return: sex: płeć
       :rtype: str
       """
        return self.sex

    # Zapisz dane pacjenta w słowniku
    def data_to_dictionary(self):
        """
        Funkcja zwracająca dane pacjenta w postaci słownika
        :return: data: dane pacjenta
        :rtype: dict
        """
        data = {'Patient number': self.number, 'Sex': self.sex, 'Age': self.age}
        return data
