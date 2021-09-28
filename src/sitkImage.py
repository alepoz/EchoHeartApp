import SimpleITK as sitk
import numpy as np
import cv2.cv2 as cv


class SitkImage(object):

    def __init__(self, image, data):
        # Dane obrazu
        self.patient_num = data['PatientNumber']
        self.name = data['Name']
        self.projection = data['Projection']
        self.phase = data['Phase']
        self.img_type = data['ImageType']
        self.img_quality = data['ImageQuality']

        # Parametry obrazu
        self.sitk_image = image
        self.img_ndarray = sitk.GetArrayFromImage(self.sitk_image)
        self.img_array = np.array(self.img_ndarray[0])
        self.img_out = self.image_to_uint8(self.img_array)
        self.height = self.img_array.shape[0]
        self.width = self.img_array.shape[1]

        # Parametry komory
        self.apex_location = None
        self.base_location = None
        self.base_length = None
        self.long_axis_length = None
        self.short_axis_length = None
        self.epicardial_length = None
        self.endocardial_length = None
        self.myocardial_length = None
        self.epicardial_contour = None
        self.endocardial_contour = None
        self.myocardial_contour = None
        self.gls = None
        self.segments = None
        self.ventricular_area = None
        self.ventricular_mass = None
        self.wall_area = None
        self.wall_thickness = None
        self.ventricular_volume = None
        self.ef = None

    # Settersy
    def set_img_out(self, img_out):
        self.img_out = img_out

    def set_apex_location(self, apex_location):
        self.apex_location = apex_location

    def set_base_location(self, base_location):
        self.base_location = base_location

    def set_base_length(self, base_length):
        self.base_length = base_length

    def set_long_axis_length(self, long_axis_len):
        self.long_axis_length = long_axis_len

    def set_short_axis_length(self, short_axis_len):
        self.short_axis_length = short_axis_len

    def set_epicardial_length(self, epicardial_length):
        self.epicardial_length = epicardial_length

    def set_endocardial_length(self, endocardial_length):
        self.endocardial_length = endocardial_length

    def set_myocardial_length(self, myocardial_length):
        self.myocardial_length = myocardial_length

    def set_epicardial_contour(self, cnt):
        self.epicardial_contour = cnt

    def set_endocardial_contour(self, cnt):
        self.endocardial_contour = cnt

    def set_myocardial_contour(self, cnt):
        self.myocardial_contour = cnt

    def set_ventricular_area(self, ventricular_area):
        self.ventricular_area = ventricular_area

    def set_wall_area(self, wall_area):
        self.wall_area = wall_area

    def set_ventricular_volume(self, ventricular_volume):
        self.ventricular_volume = ventricular_volume

    def set_ventricular_mass(self, ven_mass):
        self.ventricular_mass = ven_mass

    def set_segments(self, segments):
        self.segments = segments

    def set_wall_thickness(self, wall_thickness):
        self.wall_thickness = wall_thickness

    def set_gls(self, gls):
        self.gls = gls

    def set_ef(self, ef):
        self.ef = ef

    # Gettersy
    def get_name(self):
        return self.name

    def get_projection(self):
        return self.projection

    def get_phase(self):
        return self.phase

    def get_img_type(self):
        return self.img_type

    def get_quality(self):
        return self.img_quality

    def get_img_sitk(self):
        return self.sitk_image

    def get_img_ndarray(self):
        return self.img_ndarray

    def get_img_array(self):
        return self.img_array

    def get_img_out(self):
        return self.img_out

    def get_apex_location(self):
        return self.apex_location

    def get_base_location(self):
        return self.base_location

    def get_base_length(self):
        return self.base_length

    def get_long_axis_length(self):
        return self.long_axis_length

    def get_short_axis_length(self):
        return self.short_axis_length

    def get_epicardial_length(self):
        return self.epicardial_length

    def get_endocardial_length(self):
        return self.endocardial_length

    def get_myocardial_length(self):
        return self.myocardial_length

    def get_epicardial_contour(self):
        return self.epicardial_contour

    def get_endocardial_contour(self):
        return self.endocardial_contour

    def get_myocardial_contour(self):
        return self.myocardial_contour

    def get_ventricular_area(self):
        return self.ventricular_area

    def get_wall_area(self):
        return self.wall_area

    def get_ventricular_volume(self):
        return self.ventricular_volume

    def get_ventricular_mass(self):
        return self.ventricular_mass

    def get_segments(self):
        return self.segments

    def get_wall_thickness(self):
        return self.wall_thickness

    def get_gls(self):
        return self.gls

    def get_ef(self):
        return self.ef

    # Reprezentacja obrazu typu uint8
    def image_to_uint8(self, original_image):
        """
        Zwraca obraz o wartościach pikseli dostosowanych do skali 8bitowej
        :param original_image:
        :type original_image: numpy.array
        :return: img_out: obraz wyjściowy
        :rtype: numpy.array
        """
        img_out = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
        if self.img_type == 'Mask':
            img_out[img_out[:, :] == 3] = 255
            img_out[img_out[:, :] == 2] = int(255 * 2 / 3)
            img_out[img_out[:, :] == 1] = int(255 / 3)
        return img_out

    # Przywrócenie oryginalnego obrazu
    def reset_out_image(self):
        """
        Resetuje obraz wyjściowy (wyświetlany) do stanu pierwotnego
        :return: None
        """
        self.img_out = self.image_to_uint8(self.img_array)

    # Zapisz dane w słowniku
    def data_to_dictionary(self):
        """
        Zwraca dane obrazu w postaci słownika
        :return: data: dane obrazu
        :rtype: dict
        """
        data = {'Projection': self.projection, 'Phase': self.phase, 'Image type': self.img_type,
                'Image quality': self.img_quality, 'Apex location': self.apex_location,
                'Base location': self.base_location, 'Base length [cm]': self.base_length,
                'Long axis length [cm]': self.long_axis_length, 'Short axis length [cm]': self.short_axis_length,
                'Epicardial line length [cm]': self.epicardial_length,
                'Myocardial line length [cm]': self.myocardial_length,
                'Endocardial line length [cm]': self.endocardial_length, 'GLS [%]': self.gls,
                'Ventricular area [cm^2]': self.ventricular_area, 'Ventricular volume [cm^3]': self.ventricular_volume,
                'Ventricular mass [g]': self.ventricular_mass, 'EF': self.ef, 'Wall area [cm^2]': self.wall_area,
                'Average wall thickness [cm]': self.wall_thickness, 'Segments': self.segments}

        return data
