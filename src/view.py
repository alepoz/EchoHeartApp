from tkinter.ttk import Combobox
import matplotlib

matplotlib.use("TkAgg")
import tkinter as tk
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pylab as plt


class View(object):

    def __init__(self, controller):
        self.controller = controller
        self.window = tk.Tk()
        self.window.geometry("1100x950")  # wymiary okna
        self.window.title("EchocardiographyApp")  # tytuł aplikacji
        self.f_top = Frame(self.window)  # górna ramka
        self.f_top.pack(side=TOP)
        self.f_bottom = Frame(self.window)  # dolna ramka
        self.f_bottom.pack(side=BOTTOM)
        self.f_left = Frame(self.f_top)  # lewa górna ramka
        self.f_left.pack(side=LEFT)
        self.f_right = Frame(self.f_top)  # prawa górna ramka
        self.f_right.pack(side=RIGHT)
        self.f_right_top = Frame(self.f_right)  # ramka wyświetlania obrazu
        self.f_right_top.pack(side=TOP)

        # Panel wyświetlania obrazu
        self.fig = plt.figure(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.f_right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.f_right)  # pasek narzędzi
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Pasek menu
        self.menubar = tk.Menu(self.window)
        self.cascade = tk.Menu(self.menubar)  # tworzenie 1. kaskady menu
        self.cascade2 = tk.Menu(self.menubar)  # tworzenie 2. kaskady menu
        self.menubar.add_cascade(label="Wczytaj", menu=self.cascade)  # dodanie 1. kaskady do menu głównego
        self.menubar.add_cascade(label="Zapisz dane", menu=self.cascade2)  # dodanie 2. kaskady do menu głównego
        self.cascade.add_command(label="Wybierz folder",
                                 command=self.controller.choose_patients_folder)  # dodanie opcji do podmenu
        self.cascade2.add_command(label="Aktualnego obrazu",
                                  command=self.controller.save_data_to_json_file)  # dodanie opcji do podmenu
        self.cascade2.add_command(label="Wszystkich obrazów",
                                  command=self.controller.save_all_data_to_json_file)
        self.menubar.add_command(label="Zamknij aplikację", command=self.controller.exit_app)  # dodanie opcji

        self.window.config(menu=self.menubar)

        # Przewijanie obrazów
        self.f_right_bot = LabelFrame(self.f_right, bd=0, relief=RAISED, height=20, width=100)
        self.f_right_bot.pack(side=BOTTOM)
        self.button_prev = Button(self.f_right_bot, text="Poprzedni obraz", state=DISABLED,
                                  command=self.controller.previous_image)
        self.button_next = Button(self.f_right_bot, text="Następny obraz", state=DISABLED,
                                  command=self.controller.next_image)
        self.button_reset = Button(self.f_right_bot, text="Resetuj obraz", state=DISABLED,
                                   command=self.controller.reset_image)
        self.button_prev.grid(row=0, column=0, padx=10, pady=5)
        self.button_next.grid(row=0, column=2, padx=10, pady=5)
        self.button_reset.grid(row=0, column=4, padx=10, pady=5)

        # Lewa ramka
        self.f_left_top = LabelFrame(self.f_left, text="Dane pacjenta", bd=1, relief=RAISED, height=80, width=100)
        self.f_left_top.grid(sticky='W', padx=2, pady=5, ipadx=2, ipady=5)
        self.f_left_mid = LabelFrame(self.f_left, text="Dane obrazu", bd=1, relief=RAISED, height=80, width=100)
        self.f_left_mid.grid(sticky='W', padx=2, pady=5, ipadx=2, ipady=5)
        self.f_left_bot = LabelFrame(self.f_left, text="Parametry komory", bd=1, relief=RAISED, height=80, width=100)
        # self.f_left_bot.pack(side=BOTTOM)
        self.f_left_bot.grid(sticky='W', padx=2, pady=5, ipadx=2, ipady=5)

        # Lewa górna ramka - panel górny
        self.l_num_pat = Label(self.f_left_top, height=1, width=15, text="Numer pacjenta:", anchor='w')
        self.l_num_pat.grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.l_sex = Label(self.f_left_top, height=1, width=15, text="Płeć pacjenta:", anchor='w')
        self.l_sex.grid(row=1, column=0, sticky=W, padx=5, pady=5)
        self.l_age = Label(self.f_left_top, height=1, width=15, text="Wiek pacjenta:", anchor='w')
        self.l_age.grid(row=2, column=0, sticky=W, padx=5, pady=5)
        # Wyświetlanie danych
        self.t_num_pat = Label(self.f_left_top, height=1, width=15, text='', justify='left')
        self.t_num_pat.grid(row=0, column=1, sticky=W, padx=5, pady=5)
        self.t_sex = Label(self.f_left_top, height=1, width=15, text='', justify='left')
        self.t_sex.grid(row=1, column=1, sticky=W, padx=5, pady=5)
        self.t_age = Label(self.f_left_top, height=1, width=15, text='', justify='left')
        self.t_age.grid(row=2, column=1, sticky=W, padx=5, pady=5)

        # Lewa górna ramka - panel środkowy
        self.l_projection = Label(self.f_left_mid, height=1, width=15, text="Typ projekcji:", anchor='w')
        self.l_projection.grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.l_phase = Label(self.f_left_mid, height=1, width=15, text="Faza:", anchor='w')
        self.l_phase.grid(row=1, column=0, sticky=W, padx=5, pady=5)
        self.l_img_type = Label(self.f_left_mid, height=1, width=15, text="Typ obrazu:", anchor='w')
        self.l_img_type.grid(row=2, column=0, sticky=W, padx=5, pady=5)
        self.l_quality = Label(self.f_left_mid, height=1, width=15, text="Jakość obrazu:", anchor='w')
        self.l_quality.grid(row=3, column=0, sticky=W, padx=5, pady=5)
        # Wyświetlanie danych
        self.t_projection = Label(self.f_left_mid, height=1, width=15, text='', justify='left')
        self.t_projection.grid(row=0, column=1, sticky=W, padx=5, pady=5)
        self.t_phase = Label(self.f_left_mid, height=1, width=15, text='', justify='left')
        self.t_phase.grid(row=1, column=1, sticky=W, padx=5, pady=5)
        self.t_img_type = Label(self.f_left_mid, height=1, width=15, text='', justify='left')
        self.t_img_type.grid(row=2, column=1, sticky=W, padx=5, pady=5)
        self.t_quality = Label(self.f_left_mid, height=1, width=15, text='', justify='left')
        self.t_quality.grid(row=3, column=1, sticky=W, padx=5, pady=5)

        # Lewa górna ramka - panel dolny
        # Parametry
        self.var1 = BooleanVar()
        self.var2 = BooleanVar()
        self.var3 = BooleanVar()
        self.var4 = BooleanVar()
        self.var5 = BooleanVar()
        self.var6 = BooleanVar()
        self.var7 = BooleanVar()
        self.var8 = BooleanVar()
        self.var9 = BooleanVar()
        self.var10 = BooleanVar()
        self.var11 = BooleanVar()
        self.var12 = BooleanVar()
        self.var13 = BooleanVar()
        self.var14 = BooleanVar()

        self.check_butt1 = Checkbutton(self.f_left_bot, text='Podstawa komory', anchor='w', variable=self.var1,
                                       onvalue=1, offvalue=0, height=1, width=20)
        self.check_butt1.grid(row=0, column=0, sticky=W)

        self.check_butt2 = Checkbutton(self.f_left_bot, text='Koniuszek komory', anchor='w', variable=self.var2,
                                       onvalue=1, offvalue=0, height=1, width=20)
        self.check_butt2.grid(row=1, column=0, sticky=W)

        self.check_butt3 = Checkbutton(self.f_left_bot, text='Osie komory', anchor='w', variable=self.var3, onvalue=1,
                                       offvalue=0, height=1, width=20)
        self.check_butt3.grid(row=2, column=0, sticky=W)

        self.check_butt4 = Checkbutton(self.f_left_bot, text='Objętość komory', anchor='w', variable=self.var4,
                                       onvalue=1,
                                       offvalue=0, height=1, width=20)
        self.check_butt4.grid(row=3, column=0, sticky=W)

        self.check_butt5 = Checkbutton(self.f_left_bot, text='Frakcja wyrzutowa', anchor='w',
                                       variable=self.var5, onvalue=1, offvalue=0, height=1, width=20)
        self.check_butt5.grid(row=4, column=0, sticky=W)

        self.check_butt6 = Checkbutton(self.f_left_bot, text='Masa komory', anchor='w', variable=self.var6,
                                       onvalue=1, offvalue=0, height=1, width=20)
        self.check_butt6.grid(row=5, column=0, sticky=W)

        self.check_butt7 = Checkbutton(self.f_left_bot, text='Linie ściany', anchor='w', variable=self.var7, onvalue=1,
                                       offvalue=0, height=1, width=20)
        self.check_butt7.grid(row=6, column=0, sticky=W)

        self.check_butt8 = Checkbutton(self.f_left_bot, text='Współczynnik GLS', anchor='w',
                                       variable=self.var8, onvalue=1, offvalue=0, height=1, width=20)
        self.check_butt8.grid(row=7, column=0, sticky=W)

        self.l_segmentation1 = Label(self.f_left_bot, height=2, width=30,
                                     text="Segmentacja ściany - metoda 1.\n(wg długości linii ściany):", anchor='w')
        self.l_segmentation1.grid(row=0, column=1, sticky=W, padx=2, pady=5)

        self.check_butt9 = Checkbutton(self.f_left_bot, text='Segmentacja 16-elementowa', anchor='w',
                                       variable=self.var9, onvalue=1, offvalue=0, height=1, width=30)
        self.check_butt9.grid(row=1, column=1, sticky=W)

        self.check_butt10 = Checkbutton(self.f_left_bot, text='Segmentacja 17-elementowa', anchor='w',
                                        variable=self.var10, onvalue=1, offvalue=0, height=1, width=30)
        self.check_butt10.grid(row=2, column=1, sticky=W)

        self.l_segmentation2 = Label(self.f_left_bot, height=2, width=26,
                                     text="Segmentacja  ściany - metoda 2.\n(wg linii ortogonalnych)", anchor='w')
        self.l_segmentation2.grid(row=3, column=1, sticky=W, padx=2, pady=5)

        self.check_butt11 = Checkbutton(self.f_left_bot, text='Segmentacja 16-elementowa', anchor='w',
                                        variable=self.var11, onvalue=1, offvalue=0, height=1, width=30)
        self.check_butt11.grid(row=4, column=1, sticky=W)

        self.check_butt12 = Checkbutton(self.f_left_bot, text='Segmentacja 17-elementowa', anchor='w',
                                        variable=self.var12, onvalue=1, offvalue=0, height=1, width=30)
        self.check_butt12.grid(row=5, column=1, sticky=W)

        self.check_butt13 = Checkbutton(self.f_left_bot, text='Powierzchnia przekroju komory', anchor='w',
                                        variable=self.var13, onvalue=1, offvalue=0, height=1, width=30)
        self.check_butt13.grid(row=6, column=1, sticky=W)

        self.check_butt14 = Checkbutton(self.f_left_bot, text='Powierzchnia przekroju ściany', anchor='w',
                                        variable=self.var14, onvalue=1, offvalue=0, height=1, width=30)
        self.check_butt14.grid(row=7, column=1, sticky=W)

        # Przycisk "wyznacz"
        self.conf_button = Button(self.f_left_bot, text='Wyznacz', state=DISABLED,
                                  command=self.controller.appoint_parameters)
        self.conf_button.grid(row=11, column=0, sticky=W, pady=5, padx=5)

        # Dolna ramka
        self.fl_bottom = LabelFrame(self.f_bottom, text="Wyznaczone parametry", bd=1, relief=RAISED, height=70,
                                    width=200)
        self.fl_bottom.grid(sticky='N', pady=8)
        self.l_base_length = Label(self.fl_bottom, height=1, width=26, text="Długość podstawy [cm]:", anchor='w')
        self.l_base_length.grid(row=0, column=0, sticky=W, padx=2, pady=5)
        self.l_long_axis_length = Label(self.fl_bottom, height=1, width=26, text="Długość osi długiej [cm]:",
                                        anchor='w')
        self.l_long_axis_length.grid(row=1, column=0, sticky=W, padx=2, pady=5)
        self.l_short_axis_length = Label(self.fl_bottom, height=1, width=26, text="Długość osi krótkiej [cm]:",
                                         anchor='w')
        self.l_short_axis_length.grid(row=2, column=0, sticky=W, padx=2, pady=5)
        self.l_ven_volume = Label(self.fl_bottom, height=1, width=26, text="Objętość komory [cm^3]:", anchor='w')
        self.l_ven_volume.grid(row=3, column=0, sticky=W, padx=2, pady=5)
        self.l_ven_area = Label(self.fl_bottom, height=1, width=26, text="Pole przekroju komory [cm^2]:", anchor='w')
        self.l_ven_area.grid(row=4, column=0, sticky=W, padx=2, pady=5)
        self.l_ef = Label(self.fl_bottom, height=1, width=26, text="Frankcja wyrzutowa [%]:", anchor='w')
        self.l_ef.grid(row=5, column=0, sticky=W, padx=2, pady=5)
        self.l_mass = Label(self.fl_bottom, height=1, width=26, text="Masa komory [g]", anchor='w')
        self.l_mass.grid(row=6, column=0, sticky=W, padx=2, pady=5)

        self.l_epi_length = Label(self.fl_bottom, height=1, width=26, text="Długość linii epicardialnej [cm]:",
                                  anchor='w')
        self.l_epi_length.grid(row=0, column=2, sticky=W, padx=5, pady=5)
        self.l_myo_length = Label(self.fl_bottom, height=1, width=26, text="Długość linii środkowej [cm]:", anchor='w')
        self.l_myo_length.grid(row=1, column=2, sticky=W, padx=5, pady=5)
        self.l_endo_length = Label(self.fl_bottom, height=1, width=26, text="Długość linii endocardialnej [cm]:",
                                   anchor='w')
        self.l_endo_length.grid(row=2, column=2, sticky=W, padx=5, pady=5)
        self.l_wall_thickness = Label(self.fl_bottom, height=1, width=26, text="Średnia grubość ściany [cm]:",
                                      anchor='w')
        self.l_wall_thickness.grid(row=3, column=2, sticky=W, padx=5, pady=5)
        self.l_wall_area = Label(self.fl_bottom, height=1, width=26, text="Pole przekroju ściany [cm^2]:", anchor='w')
        self.l_wall_area.grid(row=4, column=2, sticky=W, padx=5, pady=5)
        self.l_gls = Label(self.fl_bottom, height=1, width=26, text="GLS [%]:", anchor='w')
        self.l_gls.grid(row=5, column=2, sticky=W, padx=5, pady=5)

        # Dane do wpisania
        self.t_base_length = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_base_length.grid(row=0, column=1, sticky=W, padx=5, pady=5)
        self.t_long_axis_length = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_long_axis_length.grid(row=1, column=1, sticky=W, padx=5, pady=5)
        self.t_short_axis_length = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_short_axis_length.grid(row=2, column=1, sticky=W, padx=5, pady=5)
        self.t_ven_volume = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_ven_volume.grid(row=3, column=1, sticky=W, padx=5, pady=5)
        self.t_ven_area = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_ven_area.grid(row=4, column=1, sticky=W, padx=5, pady=5)
        self.t_ef = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_ef.grid(row=5, column=1, sticky=W, padx=5, pady=5)
        self.t_mass = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_mass.grid(row=6, column=1, sticky=W, padx=5, pady=5)

        self.t_epi_length = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_epi_length.grid(row=0, column=3, sticky=W, padx=5, pady=5)
        self.t_myo_length = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_myo_length.grid(row=1, column=3, sticky=W, padx=5, pady=5)
        self.t_endo_length = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_endo_length.grid(row=2, column=3, sticky=W, padx=5, pady=5)
        self.t_wall_thickness = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_wall_thickness.grid(row=3, column=3, sticky=W, padx=5, pady=5)
        self.t_wall_area = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_wall_area.grid(row=4, column=3, sticky=W, padx=5, pady=5)
        self.t_gls = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_gls.grid(row=5, column=3, sticky=W, padx=5, pady=5)

        self.cb_value = StringVar()  # zmienna typu StringVar, która zostanie podpięta pod kontrolkę Combobox
        self.cb_segment = Combobox(self.fl_bottom, height=1, width=20, state=DISABLED,
                                   textvariable=self.cb_value)  # tworzenie kontrolki Combobox
        self.cb_segment.grid(row=0, column=4, sticky='W', padx=5, pady=5)
        # domyślne nazwy segmentów w Combox
        self.cb_segment['values'] = (
            'Apex', 'Apical septal', 'Mid inferoseptal', 'Basal inferoseptal', 'Apical lateral',
            'Mid anterolateral', 'Basal anterolateral')

        # ustawienie elementów zawartych na liście rozwijanej
        self.cb_segment.current(0)  # ustawienie domyślnego indeksu zaznaczenia
        self.cb_segment.bind("<<ComboboxSelected>>",
                             self.controller.change_segment)  # podpięcie metody pod zdarzenie zmiany zaznaczenia

        self.l_seg_epi_len = Label(self.fl_bottom, height=1, width=26, text="Długość linii epicardialnej [cm]:",
                                   anchor='w')
        self.l_seg_epi_len.grid(row=1, column=4, sticky=W, padx=5, pady=5)
        self.l_seg_myo_len = Label(self.fl_bottom, height=1, width=30, text="Długość linii środkowej [cm]:", anchor='w')
        self.l_seg_myo_len.grid(row=2, column=4, sticky=W, padx=5, pady=5)
        self.l_seg_endo_len = Label(self.fl_bottom, height=1, width=30, text="Długość linii endocardialnej [cm]:",
                                    anchor='w')
        self.l_seg_endo_len.grid(row=3, column=4, sticky=W, padx=5, pady=5)
        self.l_seg_thickness = Label(self.fl_bottom, height=1, width=30, text="Średnia grubość segmentu [cm]:",
                                     anchor='w')
        self.l_seg_thickness.grid(row=4, column=4, sticky=W, padx=5, pady=5)
        self.l_seg_area = Label(self.fl_bottom, height=1, width=30, text="Pole przekroju segmentu [cm^2]:", anchor='w')
        self.l_seg_area.grid(row=5, column=4, sticky=W, padx=5, pady=5)

        # Dane do wpisania
        self.t_seg_epi_len = Label(self.fl_bottom, bg='white', height=1, width=10, text="",
                                   anchor='w')
        self.t_seg_epi_len.grid(row=1, column=5, sticky=W, padx=5, pady=5)
        self.t_seg_myo_len = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_seg_myo_len.grid(row=2, column=5, sticky=W, padx=5, pady=5)
        self.t_seg_endo_len = Label(self.fl_bottom, bg='white', height=1, width=10, text="",
                                    anchor='w')
        self.t_seg_endo_len.grid(row=3, column=5, sticky=W, padx=5, pady=5)
        self.t_seg_thickness = Label(self.fl_bottom, bg='white', height=1, width=10, text="",
                                     anchor='w')
        self.t_seg_thickness.grid(row=4, column=5, sticky=W, padx=5, pady=5)
        self.t_seg_area = Label(self.fl_bottom, bg='white', height=1, width=10, text="", anchor='w')
        self.t_seg_area.grid(row=5, column=5, sticky=W, padx=5, pady=5)
