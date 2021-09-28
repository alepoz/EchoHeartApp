import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pylab as plt
import tkinter as tk
from tkinter import *
from tkinter import messagebox as msb, filedialog
import os
import SimpleITK as sitk
import json

# Importowanie funkcji z pakietu LVAnalysis
from Lib.LVAnalysis.base import detect_base
from Lib.LVAnalysis.apex import detect_apex
from Lib.LVAnalysis.axes import detect_axes
from Lib.LVAnalysis.wall_lines import detect_wall_lines
from Lib.LVAnalysis.wall_segmentation import wall_segmentation
from Lib.LVAnalysis.wall_segmentation_ortogonal import wall_segmentation_ortogonal
from Lib.LVAnalysis.factors import count_gls, count_ejection_fraction
from Lib.LVAnalysis.ventricular_mass import count_ventricular_mass
from Lib.LVAnalysis.section_area import count_ventricular_area, count_wall_area
from Lib.LVAnalysis.volume import count_volume

# Importowanie klasy Model oraz View
from src.model import Model
from src.view import View


class Controller(object):
    model = None
    view = None
    project_path = None

    def __init__(self):
        self.model = Model()
        self.view = View(self)

    def run(self):
        """ Wyświetla GUI aplikacji
        :return: None
        """
        self.view.window.mainloop()

    # Funkcja wyznaczająca paremetry zaznaczone w panelu "Parametry komory"
    def appoint_parameters(self):
        """ Wywołuje funkcje wyznaczające wybrane parametry
        :return: None
        """

        if any([self.view.var1.get(), self.view.var2.get(), self.view.var3.get(), self.view.var7.get(),
                self.view.var9.get(), self.view.var10.get(), self.view.var11.get(), self.view.var12.get(),
                self.view.var13.get(), self.view.var14.get()]):
            mask_name = self.get_current_img_mask_name()  # nazwa maski aktualnego obrazu
            origin_name = self.get_current_img_mask_name()[0:-3]  # nazwa oryginału aktualnego obrazu
            origin_sitk_img = self.model.get_patient().get_image_by_name(origin_name)  # obraz oryginalny
            mask_sitk_img = self.model.get_patient().get_image_by_name(mask_name)  # maska obrazu
            mask_out = mask_sitk_img.get_img_out()  # obraz wyjściowy maski
            origin_out = origin_sitk_img.get_img_out()  # obraz wyjściowy oryginału
            images_out = [origin_out, mask_out]  # lista obrazów wyjściowych

        # Podstawa
        if self.view.var1.get():
            # Funkcja detekcji bazy
            self.appoint_base(mask_sitk_img, origin_sitk_img, images_out)

        # Koniuszek
        if self.view.var2.get():
            self.appoint_apex(mask_sitk_img, origin_sitk_img, images_out)

        # Osie główne komory
        if self.view.var3.get():
            self.appoint_axes(mask_sitk_img, origin_sitk_img, images_out)

        # Objętość komory
        if self.view.var4.get():
            self.appoint_volume()

        # Współczynnik EF
        if self.view.var5.get():
            self.appoint_ef()

        # Masa komory
        if self.view.var6.get():
            self.appoint_ventricular_mass()

        # Linie ściany
        if self.view.var7.get():
            self.appoint_wall_lines(mask_sitk_img, origin_sitk_img, images_out)

        # Współczynnik GLS
        if self.view.var8.get():
            self.appoint_gls()

        # Segmentacja ściany - metoda 1.
        if self.view.var9.get() or self.view.var10.get():
            self.appoint_wall_segments(mask_sitk_img, origin_sitk_img, images_out)

        # Segmentacja ściany - metoda 2. (ortogonalnych)
        if self.view.var11.get() or self.view.var12.get():
            self.appoint_wall_segments_ortg(mask_sitk_img, origin_sitk_img, images_out)

        # Powierzchnia komory
        if self.view.var13.get():
            self.appoint_ventricular_area(mask_sitk_img, origin_sitk_img)

        # Powierzchnia ściany
        if self.view.var14.get():
            self.appoint_wall_area(mask_sitk_img, origin_sitk_img)

    # Funkcja wyświetlająca obraz
    def show_image(self, image):
        """ Wyświetla obraz w GUI
        :param image: wyświetlany obraz
        :type image: array
        :return: None
        """
        plt.gray()
        plt.subplots_adjust(0, 0.05, 0.95, 0.95, 0.1, 0.1)
        plt.imshow(image)
        plt.axis('on')
        self.view.canvas.draw()

    # Funkcja odświeżająca wyświetlany obraz
    def refresh_image(self, image):
        """ Odświeża wyświetlany obraz w GUI
        :param image: wyświetlany obraz
        :type image: array
        :return: None
        """
        self.view.fig.clf()
        plt.gray()
        plt.imshow(image)
        plt.axis('on')
        self.view.fig.canvas.draw()
        self.view.fig.canvas.flush_events()
        self.view.fig.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Funkcja wyświetlająca dane w panelach "Dane pacjenta" oraz "Dane obrazu"
    def show_image_data(self, img_index):
        """ Wyświetla dane obrazu w panelu
        :param img_index: indeks aktualnego obrazu w liście obrazów pacjenta
        :type img_index: int
        :return: None
        """
        img = self.model.get_patient().get_image_by_idx(img_index)
        self.view.t_num_pat.config(text=self.model.get_patient().get_number())
        self.view.t_sex.config(text=self.model.get_patient().get_sex())
        self.view.t_age.config(text=self.model.get_patient().get_age())
        self.view.t_projection.config(text=img.get_projection())
        self.view.t_phase.config(text=img.get_phase())
        self.view.t_img_type.config(text=img.get_img_type())
        self.view.t_quality.config(text=img.get_quality())
        self.view.conf_button.config(state="normal")

    # Funkcja zapisująca tekst z pliku CFG do słownika
    def read_txt_info(self, info_file_name):
        """
        Zapisuje tekst z pliku tekstowego CFG do słownika
        :param info_file_name: Ścieżka pliku CFG
        :type: string
        :return: info: Słownik z danymi pacjenta
        :rtype: dict
        """
        info = {}
        with open(info_file_name, 'r') as info_file:
            info_file_txt = info_file.readlines()  # wczytywanie wszystkich linii z pliku tekstowego
        for line in info_file_txt:
            line = line.split(':')  # podział tekstu pojedyńczej linii znakiem ":" na 2 części
            try:
                # zapis 1. część linii jako słowo klucz w słowniku i 2. części jako jego wartość
                info[line[0]] = int(line[1][1:-1])  # spróbuj zamienić tekst na wartość typu int
            except ValueError:
                try:
                    info[line[0]] = float(line[1][1:-1])  # spróbuj zamienić tekst na wartość typu float
                except ValueError:
                    info[line[0]] = line[1][1:-1]  # zapisz tekst jako string
        return info

    # Funkcja zapisująca dane z nazwy pliku MHD do słownika
    def load_image_txt_data(self, txt_info, img_filename):
        """ Funkcja wczytująca dane obrazu z nazwy pliku MHD obrazu
        :param txt_info: Dane z pliku tekstowego CFG
        :type txt_info: dict
        :param img_filename: Nazwa pliku MHD
        :type img_filename: str
        :return: txt_data: Słownik z danymi pacjenta i obrazu
        :rtype: dict
        """
        txt_data = {}
        sfilename = img_filename.split("_")
        txt_data['Name'] = img_filename[0:-4]  # nazwa pliku (bez rozszerzenia)
        if txt_data['Name'][-2::] == 'gt':  # rodzaj obrazu: (oryginalny lub jego maska)
            txt_data['ImageType'] = 'Mask'
        else:
            txt_data['ImageType'] = 'Original'
        txt_data['PatientNumber'] = sfilename[0][7::]  # numer pacjenta
        txt_data['Projection'] = sfilename[1]  # projekcja (2CH lub 4CH)
        txt_data['Phase'] = sfilename[2][0:2]  # faza (ES lub ED)
        txt_data.update(txt_info)  # aktualizacja słownika o dane z nazwy pliku

        return txt_data

    def choose_patients_folder(self):
        """ Uruchamia panel wyboru folderu pacjenta, a następnie wczytuje do aplikacji wszystkie dane
        dotyczące pacjenta
        :return: None
        """
        try:
            # otwieranie systemowego okna wyboru folderu
            input_folder = filedialog.askdirectory(title=u'Wybierz folder')
            files = os.listdir(input_folder)  # lista wszystkich plików w wybranym folderze
            images_dictionary = {}
            txt_info = None
            for file in files:
                if file.endswith('.cfg'):
                    # zapisz dane pliku tekstowego do słownika txt_info
                    txt_info = self.read_txt_info(input_folder + "//" + file)
                    break
            if txt_info:
                # zapisz numer pacjenta do słownika (na podstawie nazwy folderu)
                txt_info['PatientNumber'] = input_folder.split('/')[-1][7::]
            for file in files:
                if file.endswith('.mhd') and not 'sequence' in file:
                    # wczytaj plik MHD za pomocą biblioteki SimpleITK
                    images_dictionary[file] = sitk.ReadImage(input_folder + "//" + file)
            # wskaż 1. obraz z wczytanych jako aktualny do wyświetlenia
            self.model.current_img_index = 0
        except:
            msb.showerror("Info", "Nie wybrano folderu lub folder jest niewłaściwy")
            return

        # Wczytwanie danych z pliku tekstowego
        try:
            # dodaj pacjenta
            self.model.add_patient(txt_info)
            images_names_list = list(images_dictionary.keys())
            # dla każdego pliku zapisz informacje o obrazie z nazwy pliku
            for i, img in enumerate(list(images_dictionary.values())):
                txt_data = self.load_image_txt_data(txt_info, images_names_list[i])
                # dodaj obraz do pacjenta
                self.model.get_patient().add_image(img, txt_data)
        except:
            msb.showerror("Info", "Błąd wczytywania danych wybranego pacjenta")
            return

        # Wyświetlneie danych aktualnego obrazu
        try:
            self.show_image_data(self.model.current_img_index)  # wyświetl dane obrazu (pierwszy z listy wczytanych)
        except:
            msb.showerror("Info", "Błąd wczytywania informacji o danym obrazie")
            return

        # Wyświetlenie aktualnego obrazu
        try:
            self.view.button_reset.config(state="normal")
            # Funkcja sprawdza ilość wczytanych obrazów, w przypadku l. mnogiej odblokowuje przcisk "Następny"
            if len(self.model.get_patient().get_images()) > 1:
                self.view.button_next.config(state="normal")
            # wyświetl obraz (pierwszy z listy wczytanych)
            self.show_image(
                self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())
        except:
            msb.showerror("Info", "Błąd wyświetlania danego obrazu")
            return

    # Wyświetl dane dotyczące poprzedniego obrazu
    def previous_image(self):
        """Wczytuje i wyświetla poprzedni obraz w liście oraz jego dane
        :return: None
        """
        self.model.current_img_index = self.model.current_img_index - 1
        self.show_image(self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())
        self.show_image_data(self.model.current_img_index)
        self.show_parameters()
        self.change_combobox_values(self.view.t_projection.cget('text'))
        if self.model.current_img_index == 0:
            self.view.button_prev.config(state=tk.DISABLED)
        else:
            self.view.button_prev.config(state=tk.NORMAL)
        if self.model.current_img_index == len(self.model.get_patient().get_images()) - 1:
            self.view.button_next.config(state=tk.DISABLED)
        else:
            self.view.button_next.config(state=tk.NORMAL)

    # Wyświetl dane dotyczące następnego obrazu
    def next_image(self):
        """ Wczytuje i wyświetla  następny obraz w liście oraz jego dane
        :return: None
        """
        self.model.current_img_index = self.model.current_img_index + 1
        self.show_image(self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())
        self.show_image_data(self.model.current_img_index)
        self.show_parameters()
        self.change_combobox_values(self.view.t_projection.cget('text'))
        if self.model.current_img_index == 0:
            self.view.button_prev.config(state=tk.DISABLED)
        else:
            self.view.button_prev.config(state=tk.NORMAL)
        if self.model.current_img_index == len(self.model.get_patient().get_images()) - 1:
            self.view.button_next.config(state=tk.DISABLED)
        else:
            self.view.button_next.config(state=tk.NORMAL)

    # Nazwa maski aktualnie wyświetlanego obrazu
    def get_current_img_mask_name(self):
        """ Zwraca nazwę maski aktualnie wyświetlanego obrazu
        :return: name: nazwa obrazu
        """
        name = 'patient' + self.view.t_num_pat.cget("text") + '_' + self.view.t_projection.cget("text") + '_' + \
               self.view.t_phase.cget("text") + '_gt'
        return name

    # Resetuj obraz (usuwa narysowane na obrazie parametry)
    def reset_image(self):
        """ Resetuje wyświetlany obraz (przywraca oryginalny, bez narysowanych parametrów)
        :return: None
        """
        self.model.get_patient().get_image_by_idx(self.model.current_img_index).reset_out_image()
        self.show_image(self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())

    # Wyświetl wyznaczone parametry
    def show_parameters(self):
        """ Wyświetla wyznaczone parametry ilościowe aktualnie wyświetlanego obrazu
        :return: None
        """
        # Obecnie wyświetlany obraz
        current_img = self.model.get_patient().get_image_by_idx(self.model.current_img_index)
        if current_img.get_base_length():
            self.view.t_base_length.config(text=current_img.get_base_length())
        else:
            self.view.t_base_length.config(text='')

        if current_img.get_long_axis_length():
            self.view.t_long_axis_length.config(text=current_img.get_long_axis_length())
        else:
            self.view.t_long_axis_length.config(text='')

        if current_img.get_short_axis_length():
            self.view.t_short_axis_length.config(text=current_img.get_short_axis_length())
        else:
            self.view.t_short_axis_length.config(text='')

        if current_img.get_ventricular_mass():
            self.view.t_mass.config(text=current_img.get_ventricular_mass())
        else:
            self.view.t_mass.config(text='')

        if current_img.get_epicardial_length():
            self.view.t_epi_length.config(text=current_img.get_epicardial_length())
        else:
            self.view.t_epi_length.config(text='')

        if current_img.get_endocardial_length():
            self.view.t_endo_length.config(text=current_img.get_endocardial_length())
        else:
            self.view.t_endo_length.config(text='')

        if current_img.get_myocardial_length():
            self.view.t_myo_length.config(text=current_img.get_myocardial_length())
        else:
            self.view.t_myo_length.config(text='')

        if current_img.get_gls():
            self.view.t_gls.config(text=current_img.get_gls())
        else:
            self.view.t_gls.config(text='')

        if current_img.get_ventricular_volume():
            self.view.t_ven_volume.config(text=current_img.get_ventricular_volume())
        else:
            self.view.t_ven_volume.config(text='')

        if current_img.get_ef():
            self.view.t_ef.config(text=current_img.get_ef())
        else:
            self.view.t_ef.config(text='')

        if current_img.get_segments():
            self.view.cb_segment.config(state=tk.NORMAL)
            self.show_segments()
        else:
            self.view.cb_segment.config(state=tk.DISABLED)
            self.view.t_seg_area.config(text='')
            self.view.t_seg_epi_len.config(text='')
            self.view.t_seg_myo_len.config(text='')
            self.view.t_seg_endo_len.config(text='')
            self.view.t_seg_thickness.config(text='')

        if current_img.get_wall_thickness():
            self.view.t_wall_thickness.config(text=current_img.get_wall_thickness())
        else:
            self.view.t_wall_thickness.config(text='')

        if current_img.get_ventricular_area():
            self.view.t_ven_area.config(text=current_img.get_ventricular_area())
        else:
            self.view.t_ven_area.config(text='')

        if current_img.get_wall_area():
            self.view.t_wall_area.config(text=current_img.get_wall_area())
        else:
            self.view.t_wall_area.config(text='')

    # Wyświetl nazwy segmentów
    def show_segments(self):
        """ Wyświetla nazwy segmentów w Combobox
        :return: None
        """
        if self.view.t_projection == '2CH':
            self.view.cb_segment['values'] = (
                'Apex', 'Apical inferior', 'Mid inferior', 'Basal inferior', 'Apical anterior',
                'Mid anterior', 'Basal anterior')
        if self.view.t_projection == '4CH':
            self.view.cb_segment['values'] = (
                'Apex', 'Apical septal', 'Mid inferoseptal', 'Basal inferoseptal', 'Apical lateral',
                'Mid anterolateral', 'Basal anterolateral')
        self.view.cb_segment.config(state=tk.NORMAL)

    # Wyświetl dane segmentu wybranego w liście Combobox
    def change_segment(self, event):
        """ Wyświetla wyznaczone parametry ilościowe wybranego segmentu
        :param event: Zdarzenie wywołane kliknięciem
        :return: None
        """
        current_img = self.model.get_patient().get_image_by_idx(self.model.current_img_index)
        found = False
        for name, segment in current_img.get_segments().items():
            if name == self.view.cb_value.get():
                found = True
                self.view.t_seg_area.config(text=segment['Area [cm^2]'])
                self.view.t_seg_epi_len.config(text=segment['Epicardial line length [cm]'])
                self.view.t_seg_myo_len.config(text=segment['Myocardial line length [cm]'])
                self.view.t_seg_endo_len.config(text=segment['Endocardial line length [cm]'])
                self.view.t_seg_thickness.config(text=segment['Thickness [cm]'])
        if not found:
            self.view.t_seg_area.config(text='')
            self.view.t_seg_epi_len.config(text='')
            self.view.t_seg_myo_len.config(text='')
            self.view.t_seg_endo_len.config(text='')
            self.view.t_seg_thickness.config(text='')

    # Zmień nazwy segmentów w liście Combobox (w zależności od projekcji)
    def change_combobox_values(self, view):
        """ Funkcja zmienia wyświetlane nazwy segmentów w liście rozwijanej Combobox
        :param view: projekcja aktualnie wyświetlanego obrazu
        :type view: str
        :return: None
        """
        if view == '2CH':
            self.view.cb_segment['values'] = ('Apex', 'Apical inferior', 'Mid inferior', 'Basal inferior',
                                              'Apical anterior', 'Mid anterior', 'Basal anterior')
        if view == '4CH':
            self.view.cb_segment['values'] = (
                'Apex', 'Apical septal', 'Mid inferoseptal', 'Basal inferoseptal', 'Apical lateral',
                'Mid anterolateral', 'Basal anterolateral')

    # Wyznacz podstawę
    def appoint_base(self, mask_sitk_img, origin_sitk_img, images_out):
        '''
        Funkcja wyznaczająca podstawę na aktualnym obrazie i wyświetlająca otrzymane dane
        :param mask_sitk_img: maska aktualnie wyświetlanego obrazu
        :param origin_sitk_img: obraz USG wyświetlanego obrazu
        :param images_out: obrazy wyjściowe
        :return: None
        '''
        images_out, mid_px, px1, px2, length = detect_base(mask_sitk_img.get_img_array(), images_out)
        # Przypisanie przetworzonego obrazu wjściowego obiektowi klasy SitkImage
        if len(images_out) == 2:
            origin_sitk_img.set_img_out(images_out[0])
            mask_sitk_img.set_img_out(images_out[1])
        # Zapis danych położenia punktów bazy w obiekcie klasy SitkImage
        mask_sitk_img.set_base_location((mid_px, px1, px2))
        origin_sitk_img.set_base_location((mid_px, px1, px2))
        mask_sitk_img.set_base_length(length)
        origin_sitk_img.set_base_length(length)
        # Odświeżnie wyświetlanego obrazu po przetworzeniu
        self.refresh_image(self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())

    # Wyznacz koniuszek
    def appoint_apex(self, mask_sitk_img, origin_sitk_img, images_out):
        """ Funkcja wyznaczająca punkt koniuszka na aktualnym obrazie i wyświetlająca otrzymane dane
        :param mask_sitk_img: maska aktualnie wyświetlanego obrazu
        :type mask_sitk_img: array
        :param origin_sitk_img: obraz USG wyświetlanego obrazu
        :type origin_sitk_img: array
        :param images_out: obrazy wyjściowe
        :type images_out: list
        :return None
        """
        if not mask_sitk_img.get_base_location():
            msb.showerror("Info", "Brakuje wyznaczonej bazy")
            return

        images_out, apex = detect_apex(mask_sitk_img.get_img_array(),
                                       images_out, mask_sitk_img.get_base_location()[0])
        if len(images_out) == 2:
            origin_sitk_img.set_img_out(images_out[0])
            mask_sitk_img.set_img_out(images_out[1])
            # Zapis danych położenia punktu koniuszka w obiekcie klasy SitkImage
            origin_sitk_img.set_apex_location(apex)
            mask_sitk_img.set_apex_location(apex)
        # Odświeżnie wyświetlanego obrazu po przetworzeniu
        self.refresh_image(
            self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())

    # Wyznacz osie główne komory
    def appoint_axes(self, mask_sitk_img, origin_sitk_img, images_out):
        """
            Funkcja wyznaczająca osie komory na aktualnym obrazie i wyświetlająca otrzymane dane
            :param mask_sitk_img: maska aktualnie wyświetlanego obrazu
            :type mask_sitk_img: array
            :param origin_sitk_img: obraz USG wyświetlanego obrazu
            :type origin_sitk_img: array
            :param images_out: obrazy wyjściowe
            :type images_out: list
            :return: None
        """
        if not mask_sitk_img.get_base_location() or not mask_sitk_img.get_apex_location():
            msb.showerror("Info", "Brakuje wyznaczonej bazy lub koniuszka")
            return
        images_out, long_len, short_len = detect_axes(mask_sitk_img.get_img_array(), images_out,
                                                      mask_sitk_img.get_base_location()[0],
                                                      mask_sitk_img.get_apex_location())
        if len(images_out) == 2:
            origin_sitk_img.set_img_out(images_out[0])
            mask_sitk_img.set_img_out(images_out[1])
            # Zapis danych długości osi w obiekcie klasy SitkImage
            origin_sitk_img.set_long_axis_length(long_len)
            mask_sitk_img.set_long_axis_length(long_len)
            origin_sitk_img.set_short_axis_length(short_len)
            mask_sitk_img.set_short_axis_length(short_len)
        self.show_parameters()
        # Odświeżnie wyświetlanego obrazu po przetworzeniu
        self.refresh_image(
            self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())

    # Wyznacz objętość komory
    def appoint_volume(self):
        """
            Oblicza objętość komory na aktualnym obrazie i wyświetlająca wynik
            :return: None
        """
        phase = self.view.t_phase.cget('text')
        images = self.model.get_patient().get_images_by_phase(phase)
        img2CH = None
        img4CH = None
        data2CH = {}
        data4CH = {}
        for img in images:
            if img.get_projection() == '2CH' and img.get_img_type() == 'Mask':
                img2CH = img.get_img_array()
                base = img.get_base_location()
                apex = img.get_apex_location()
                if not apex:
                    msb.showerror("Info", "Brakuje wyznaczonego koniuszka w projekcji 2-jamowej")
                    return
                if not base:
                    msb.showerror("Info", "Brakuje wyznaczonej podstawy w projekcji 2-jamowej")
                    return
                data2CH['base_left'] = base[1]
                data2CH['base_mid'] = base[0]
                data2CH['base_right'] = base[2]
                data2CH['apex'] = apex

            if img.get_projection() == '4CH' and img.get_img_type() == 'Mask':
                img4CH = img.get_img_array()
                base = img.get_base_location()
                apex = img.get_apex_location()
                if not apex:
                    msb.showerror("Info", "Brakuje wyznaczonego koniuszka w projekcji 4-jamowej")
                    return
                if not base:
                    msb.showerror("Info", "Brakuje wyznaczonej podstawy w projekcji 4-jamowej")
                    return
                data4CH['base_left'] = base[1]
                data4CH['base_mid'] = base[0]
                data4CH['base_right'] = base[2]
                data4CH['apex'] = apex
                if not all(list(data2CH.values())):
                    msb.showerror("Info", "Brakuje wyznaczonej podstawy lub koniuszka")
                    return
        try:
            volume = count_volume(img2CH, img4CH, data2CH, data4CH, 'ventricular')
        except:
            msb.showerror("Info", "Błąd funkcji wyznaczania objętości")
            return

        for img in images:
            img.set_ventricular_volume(volume)

        self.show_parameters()

    # Wyznacz współczynnik frakcji wyrzutowej
    def appoint_ef(self):
        """ Oblicza współczynnik frakcji wyrzutowej aktualnego obrazu i wyświetlająca wynik
            :return: None
        """
        images = self.model.get_patient().get_images()
        es_volume = None
        ed_volume = None
        for img in images:
            if img.get_phase() == 'ES' and img.get_img_type() == 'Mask' and not es_volume:
                es_volume = img.get_ventricular_volume()
            if img.get_phase() == 'ED' and img.get_img_type() == 'Mask' and not ed_volume:
                ed_volume = img.get_ventricular_volume()

        if es_volume and ed_volume:
            ef = count_ejection_fraction(es_volume, ed_volume)
            for img in images:
                img.set_ef(ef)

        self.show_parameters()

    # Wyznacz masę komory
    def appoint_ventricular_mass(self):
        """ Oblicza masę komory na aktualnym obrazie i wyświetlająca wynik
            :return: None
        """
        images = self.model.get_patient().get_images_by_phase('ED')
        img2CH = None
        img4CH = None
        data2CH = {}
        data4CH = {}
        ven_volume = images[0].get_ventricular_volume()
        if not ven_volume:
            msb.showerror("Info", "Brakuje obliczonej objętości komory")
            return

        for img in images:
            if img.get_projection() == '2CH' and img.get_img_type() == 'Mask':
                img2CH = img.get_img_array()
                base = img.get_base_location()
                apex = img.get_apex_location()
                if not apex:
                    msb.showerror("Info", "Brakuje wyznaczonego koniuszka w projekcji 2-jamowej")
                    return
                if not base:
                    msb.showerror("Info", "Brakuje wyznaczonej podstawy w projekcji 2-jamowej")
                    return
                data2CH['base_left'] = base[1]
                data2CH['base_mid'] = base[0]
                data2CH['base_right'] = base[2]
                data2CH['apex'] = apex
            if img.get_projection() == '4CH' and img.get_img_type() == 'Mask':
                img4CH = img.get_img_array()
                base = img.get_base_location()
                apex = img.get_apex_location()
                if not apex:
                    msb.showerror("Info", "Brakuje wyznaczonego koniuszka w projekcji 4-jamowej")
                    return
                if not base:
                    msb.showerror("Info", "Brakuje wyznaczonej podstawy w projekcji 4-jamowej")
                    return
                data4CH['base_left'] = base[1]
                data4CH['base_mid'] = base[0]
                data4CH['base_right'] = base[2]
                data4CH['apex'] = apex

        # Oblicz objętość całej komory (pojemność + ściana)
        whole_volume = count_volume(img2CH, img4CH, data2CH, data4CH, 'whole')
        # Oblicz masę mięśnia komory
        mass = count_ventricular_mass(whole_volume, ven_volume)
        for img in self.model.get_patient().get_images():
            img.set_ventricular_mass(mass)

        self.show_parameters()

    # Wyznacz linie ściany
    def appoint_wall_lines(self, mask_sitk_img, origin_sitk_img, images_out):
        """ Wyznacza linie ściany na aktualnym obrazie i wyświetlająca otrzymane dane
        :param mask_sitk_img: maska aktualnie wyświetlanego obrazu
        :type mask_sitk_img: array
        :param origin_sitk_img: obraz USG wyświetlanego obrazu
        :type origin_sitk_img: array
        :param images_out: obrazy wyjściowe
        :type images_out: list
        :return: None
        """
        images_out, epi_cnt, mid_cnt, endo_cnt, epi_length, mid_length, endo_length = \
            detect_wall_lines(mask_sitk_img.get_img_array(), images_out)
        if len(images_out) == 2:
            origin_sitk_img.set_img_out(images_out[0])
            mask_sitk_img.set_img_out(images_out[1])
        # Przypisanie długości linii do obiektów SitkImage
        mask_sitk_img.set_endocardial_length(endo_length)
        origin_sitk_img.set_endocardial_length(endo_length)
        mask_sitk_img.set_myocardial_length(mid_length)
        origin_sitk_img.set_myocardial_length(mid_length)
        mask_sitk_img.set_epicardial_length(epi_length)
        origin_sitk_img.set_epicardial_length(epi_length)
        # Przypisanie konturów linii do obiektów SitkImage
        mask_sitk_img.set_endocardial_contour(endo_cnt)
        origin_sitk_img.set_endocardial_contour(endo_cnt)
        mask_sitk_img.set_myocardial_contour(mid_cnt)
        origin_sitk_img.set_myocardial_contour(mid_cnt)
        mask_sitk_img.set_epicardial_contour(epi_cnt, )
        origin_sitk_img.set_epicardial_contour(epi_cnt, )
        self.show_parameters()
        self.refresh_image(self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())

    # Wyznacz współczynnik GLS
    def appoint_gls(self):
        '''
            Funkcja obliczająca współczynnik GLS aktualnego obrazu i wyświetlająca wynik
            :return: None
        '''
        projection = self.view.t_projection.cget('text')
        images = self.model.get_patient().get_images_by_projection(projection)
        es_length = None
        ed_length = None
        for img in images:
            if img.get_phase() == 'ES' and img.get_img_type() == 'Mask':
                es_length = img.get_myocardial_length()
            if img.get_phase() == 'ED' and img.get_img_type() == 'Mask':
                ed_length = img.get_myocardial_length()

        if not es_length or not ed_length:
            msb.showerror("Info", "Brak długości linii ściany")
            return

        gls = count_gls(es_length, ed_length)
        for img in images:
            img.set_gls(gls)

        self.show_parameters()

    # Wyznacz segmenty ściany - metoda 1. (wg długości linii)
    def appoint_wall_segments(self, mask_sitk_img, origin_sitk_img, images_out):
        """
        Funkcja wyznaczająca segmenty ściany wg długości linii ściany na aktualnym obrazie
        i wyświetlająca otrzymane dane
        :param mask_sitk_img: maska aktualnie wyświetlanego obrazu
        :type mask_sitk_img: array
        :param origin_sitk_img: obraz USG wyświetlanego obrazu
        :type origin_sitk_img: array
        :param images_out: obrazy wyjściowe
        :type images_out: list
        :return: None
        """
        if self.view.var9.get():
            standard = '16-segments'
        else:
            standard = '17-segments'
        img_array = mask_sitk_img.get_img_array()
        apex = mask_sitk_img.get_apex_location()
        base = mask_sitk_img.get_base_location()
        if not apex:
            msb.showerror("Info", "Brakuje wyznaczonego koniuszka")
            return
        if not base:
            msb.showerror("Info", "Brakuje wyznaczonej podstawy")
            return
        mid_base = base[0]
        contour_epi = mask_sitk_img.get_epicardial_contour()
        contour_mid = mask_sitk_img.get_myocardial_contour()
        contour_endo = mask_sitk_img.get_endocardial_contour()
        if not contour_epi or not contour_mid or not contour_endo:
            msb.showerror("Info", "Brakuje wyznaczonych linii ściany")
            return

        projection = self.view.t_projection.cget('text')
        images_out, segments, wall_thickness = wall_segmentation(img_array, images_out, apex, mid_base,
                                                                 contour_epi, contour_mid, contour_endo,
                                                                 projection, standard)
        # Przypisz segmenty obiektom SitkImage
        origin_sitk_img.set_segments(segments)
        mask_sitk_img.set_segments(segments)
        # Przypisz grubość ściany obiektom SitkImage
        origin_sitk_img.set_wall_thickness(wall_thickness)
        mask_sitk_img.set_wall_thickness(wall_thickness)
        if len(images_out) == 2:
            origin_sitk_img.set_img_out(images_out[0])
            mask_sitk_img.set_img_out(images_out[1])
        # Odśwież wyświetlany obraz
        self.refresh_image(self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())
        self.view.cb_segment.config(state=tk.NORMAL)  # Odblokuj listę segmentów
        self.change_combobox_values(self.view.t_projection.cget('text'))  # Ustaw nazwy segmentów
        self.show_parameters()  # Pokaż parametry

    # Wyznacz segmenty ściany - metoda 2. (wg linii ortogonalnych do osi głównej)
    def appoint_wall_segments_ortg(self, mask_sitk_img, origin_sitk_img, images_out):
        """
        Funkcja wyznaczająca segmenty ściany wg ortogonalnych do osi długiej komory na aktualnym obrazie
        i wyświetlająca otrzymane dane
        :param mask_sitk_img: maska aktualnie wyświetlanego obrazu
        :type mask_sitk_img: array
        :param origin_sitk_img: obraz USG wyświetlanego obrazu
        :type origin_sitk_img: array
        :param images_out: obrazy wyjściowe
        :type images_out: list
        :return: None
        """
        if self.view.var11.get():
            standard = '16-segments'
        else:
            standard = '17-segments'
        img_array = mask_sitk_img.get_img_array()
        apex = mask_sitk_img.get_apex_location()
        base = mask_sitk_img.get_base_location()
        if not apex:
            msb.showerror("Info", "Brakuje wyznaczonego koniuszka")
            return
        if not base:
            msb.showerror("Info", "Brakuje wyznaczonej podstawy")
            return
        mid_base = base[0]
        contour_epi = mask_sitk_img.get_epicardial_contour()
        contour_mid = mask_sitk_img.get_myocardial_contour()
        contour_endo = mask_sitk_img.get_endocardial_contour()
        if not contour_epi or not contour_mid or not contour_endo:
            msb.showerror("Info", "Brakuje wyznaczonych linii ściany")
            return
        projection = self.view.t_projection.cget('text')
        try:
            images_out, segments, wall_thickness = wall_segmentation_ortogonal(img_array, images_out, apex,
                                                                               mid_base, contour_epi, contour_mid,
                                                                               contour_endo, projection, standard)
        except:
            msb.showerror("Info", "Błąd funkcji")
            return

        # Przypisz segmenty obiektom SitkImage
        origin_sitk_img.set_segments(segments)
        mask_sitk_img.set_segments(segments)
        # Przypisz grubość ściany obiektom SitkImage
        origin_sitk_img.set_wall_thickness(wall_thickness)
        mask_sitk_img.set_wall_thickness(wall_thickness)
        if len(images_out) == 2:
            origin_sitk_img.set_img_out(images_out[0])
            mask_sitk_img.set_img_out(images_out[1])
        # Odśwież wyświetlany obraz
        self.refresh_image(
            self.model.get_patient().get_image_by_idx(self.model.current_img_index).get_img_out())
        self.view.cb_segment.config(state=tk.NORMAL)  # Odblokuj listę segmentów
        self.change_combobox_values(self.view.t_projection.cget('text'))  # Ustaw nazwy segmentów
        self.show_parameters()  # Pokaż parametry

    # Wyznacz powierzchnię przekroju komory
    def appoint_ventricular_area(self, mask_sitk_img, origin_sitk_img):
        """
        Oblicza powierzchnię przekroju komory na aktualnym obrazie
        i wyświetlająca otrzymany wynik
        :param mask_sitk_img: maska aktualnie wyświetlanego obrazu
        :type mask_sitk_img: array
        :param origin_sitk_img: obraz USG wyświetlanego obrazu
        :type origin_sitk_img: array
        :return: None
        """
        area = count_ventricular_area(mask_sitk_img.get_img_array())
        mask_sitk_img.set_ventricular_area(area)
        origin_sitk_img.set_ventricular_area(area)
        self.show_parameters()

    # Wyznacz powierzchnię przekroju ściany komory
    def appoint_wall_area(self, mask_sitk_img, origin_sitk_img):
        """
        Oblicza powierzchnię przekroju ściany na aktualnym obrazie
        i wyświetlająca otrzymany wynik
        :param mask_sitk_img: maska aktualnie wyświetlanego obrazu
        :type mask_sitk_img: array
        :param origin_sitk_img: obraz USG wyświetlanego obrazu
        :type origin_sitk_img: array
        :return: None
        """
        area = count_wall_area(mask_sitk_img.get_img_array())
        mask_sitk_img.set_wall_area(area)
        origin_sitk_img.set_wall_area(area)
        self.show_parameters()

    # Zapisz dane pomiarowe pojedynczego obrazu do pliku JSON
    def save_data_to_json_file(self):
        """ Zapisuje dane pacjenta i aktualnie wyświetlanego obrazu w pliku JSON
            :return: None
        """
        # Dane w postaci słownika
        try:
            data = self.model.get_patient().data_to_dictionary()
            data_img = self.model.get_patient().get_image_by_idx(self.model.current_img_index).data_to_dictionary()
            data.update(data_img)
        except:
            msb.showerror("Info", "Brak danych")
            return
        # Zapytanie o ścieżkę zapisu
        try:
            filename = filedialog.asksaveasfilename(filetypes=[("Plik JSON", "*.json")], defaultextension="*.json")
            # Zapis danych do pliku json
            with open(filename, 'w') as json_file:
                json.dump(data, json_file, indent=2)
            json_file.close()
        except:
            msb.showerror("Info", "Nie wpisano nazwy pliku")
            return
        msb.showinfo("Info", "Dane zostały zapisane")

    # Zapisz wszystkie dane pomiarowe do pliku JSON
    def save_all_data_to_json_file(self):
        """ Zapisuje dane pacjenta i wszystkich jego obrazów do pliku JSON
            :return: None
        """
        # Dane w postaci słownika
        try:
            data = self.model.get_patient().data_to_dictionary()
            images = self.model.get_patient().get_images_by_type('Original')
            all_data = []
            for img in images:
                all_data.append(img.data_to_dictionary())
            data.update({'Images': all_data})
        except:
            msb.showerror("Info", "Brak danych")
            return
        # Zapytanie o ścieżkę zapisu
        try:
            filename = filedialog.asksaveasfilename(filetypes=[("Plik JSON", "*.json")], defaultextension="*.json")
            # Zapis danych do pliku json
            with open(filename, 'w') as json_file:
                json.dump(data, json_file, indent=2)
            json_file.close()
        except:
            msb.showerror("Info", "Wpisz nazwę pliku")
            return
        msb.showinfo("Info", "Dane zostały zapisane")

    # Zamknięcie aplikacji
    def exit_app(self):
        """ Zamyka aplikację
        :return: None
        """
        sys.exit()
