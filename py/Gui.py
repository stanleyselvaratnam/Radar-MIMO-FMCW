# Radaroc GUI PyQt5
# etienne.dubuis@hes-so.ch
# stanley.selvaratnam@master.hes-so.ch

import sys
import traceback
import math
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGroupBox, QRadioButton, QButtonGroup,QSlider,QLabel,QCheckBox,QHBoxLayout, QTabWidget, QPushButton
from PyQt5.QtCore import pyqtSignal, QThread, Qt
import iqstreamconverter   
import ratemeter            
import threading
import time
import tcpclient           
import peaks2d
from scipy.signal import find_peaks
import random
import smoothing

from scipy.signal import butter, filtfilt


class DataThread(QThread):

    update_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, server_addr=None, server_port=None, verbose=False):
        super().__init__(parent)
        self.iq_stream_converter = iqstreamconverter.IQStreamConverter(byteorder="little", signed=True, decimation=1)
        self.ratemeter = ratemeter.RateMeter(n=3)

        # thread
        self.data_lock = threading.Lock()
        self.running = True
        self.data = None
        self.metadata = None
        self.num_rx_antennas = 8    # Adaptation pour x antennes

        self.smoothing_kernel_width = 0
        self.relative_peak_height=20
        self.window="Rectangular"

        # lien TCP
        self.server_addr = server_addr
        self.server_port = server_port
        self.verbose = verbose
        self.smoothing_fir=True
        self.reconnect = False
        self.done = False

        # echelle log
        self.logscale = None                 
 
        # Parametre du chirp
        self.te=1e-6
        self.Tc=150E-6
        self.fmin=57e9
        self.fmax=66E9
        self.Nc=101

        # signal generateur
        self.R = 3
        self.V = 0
        self.A = 15

        self.phase_matrix=None
        self.fb_matrix=None
    
    # Mettre à jour distance signal synt
    def set_R(self, R):
        self.R = R  
        print(f"New R value: {self.R}")  

    # Mettre à jour vitesse signal synt
    def set_V(self, V):
        self.V = V  
        print(f"New V value: {self.V}") 

    # Mettre à jour vitesse signal synt
    def set_A(self, A):
        self.A = A  
        print(f"New A value: {self.A}°") 

    # Mettre à jour valeur smooting 
    def set_Smoothing(self, Smoothing):
        self.smoothing_kernel_width = Smoothing  
        print(f"New smoothing value: {self.smoothing_kernel_width}") 

    # 
    def smoothing_filter(self, odd=True):
        if self.smoothing_fir:
            return smoothing.SmoothingRaisedCosine(self.smoothing_kernel_width)
        else:
            return smoothing.SmoothingZeroDelayIIR(self.smoothing_kernel_width,padtype = "odd" if odd else "constant")
        
    # Mettre à jour valeur peak height 
    def set_peak_height(self, peak_height):
        self.relative_peak_height = peak_height 
        print(f"New peak_height value: {self.relative_peak_height}") 


    # Generation du signal synt
    def synthetic_data_gen(self, seq_len=150, chunk_len=200, amplitude=0.4, d=0.00263, wavelength=0.00526):
        byte_buf = b""
        seq_count = 0
        direction = True
        c = iqstreamconverter.IQConverter(byteorder="little")

        # valeur du signal FIF
        fs=1/self.te
        B=self.fmax-self.fmin
        #B=4e9
        s=B/self.Tc
        c1 = 3e8
        n=fs*self.Tc
        
        n1=0
        t = 0


        while self.running:
            # signal FIF
            tau=2*self.R/(20*c1)                # facteur 20 pour la distance
            taut=(tau+2*self.V*t/c1)
            phit = 2 * np.pi * (taut * self.fmin + s * tau * t-((s/2)*taut**2))
            # value_r1 = amplitude * math.cos(phit)#+ 0.1 * random.normalvariate())
            # value_i = amplitude * math.sin(phit)

            #tau=2*3/(20*c1)                # facteur 20 pour la distance
            #taut=(tau+2*0*t/c1)
            #phit = 2 * np.pi * (taut * self.fmin + s * tau * t-((s/2)*taut**2))
            #value_r2 = 0.05 * math.cos(phit+ 0.1 * random.normalvariate())


            # value = value_r1# + value_r2 #+ 1j * value_i     # imaginaire non necessaire pour le moment 
            
            # Calcul dynamique des déphasages en fonction de l'angle A d'une cible
            phase_shifts = self.calculate_delta_phi(self.A)
            # print(f"Phase shifts (delta_phi) for angle {self.A}°: {phase_shifts}")  # Debugging

            # Génération des signaux pour les 4 antennes
            iq_signals = [
                amplitude * math.cos(phit + shift)  # Seulement cosinus (partie réelle)
                for shift in phase_shifts
            ]

            # Conversion en binaire et stockage des 4 signaux
            for value in iq_signals:
                byte_buf += c.to_bytes(value, direction)


            # byte_buf += c.to_bytes(value, direction)
            t += 1e-6                           # periode echantillonnage
            n1 +=1

            seq_count += 1
            

            if seq_count >= seq_len:
                seq_count = 0
                direction = not direction

            if len(byte_buf) >= chunk_len:
                yield byte_buf[:chunk_len]
                byte_buf = byte_buf[chunk_len:]
                

    def set_decimation(self, decimation):
        self.iq_stream_converter.decimation = decimation

    
    def set_window(self, window):
        self.window = window

    # calcul de la distance 
    def calc_range(self, trimmed_data_np):

        # parameters essentiel
        B=self.fmax-self.fmin
        c1 = 3e8
        n=self.Tc/self.te

        mag_freq = np.abs(np.fft.fftshift(np.fft.fft(np.array(trimmed_data_np))))

        # Prendre uniquement la première antenne
        mag_freq = mag_freq[0, :] if mag_freq.ndim > 1 else mag_freq  

        if self.smoothing_kernel_width > 1:
            filter = self.smoothing_filter(True)
            mag_freq = filter.apply(mag_freq)

        # trouver les pick des FFT
        #pks, _ = find_peaks(mag_freq, height=10)

        # get peaks with relative height wrt. neighboring valleys >= self.relative_peak_height
        pks = peaks2d.peak1_best(mag_freq, relative_height_threshold=self.relative_peak_height)

        # faire correspondre index avec frequence FFT
        f_IF = [
            (index - n / 2) / (n * self.te)
            for index in pks
        ]

        # equation distance avec frequence battement
        R = [
            c1 * self.Tc * f_IF / (2*B)
            for f_IF in f_IF
        ]
        if len(R) == 0:
            print("Erreur : Aucun pic détecté pour le calcul de la distance.(R)",R)
            return []
        print("distance : (R)",R)
        return R

    # calcul de la vitesse
    def calc_Velocity(self, Matrice, n_filter=10):

        # parameters essentiel
        fs=1/self.te
        c1 = 3e8
        n=fs*self.Tc
        
        # choisir une colonne de la FFT
        colonne=Matrice[:,10]
        ligne = colonne.reshape(1, -1) 

        # trouver les pick des FFT
        #mag_freq = np.abs(np.fft.fft(trimmed_data_np)) / math.sqrt(len(trimmed_data_np))
        mag_freq = np.abs(np.fft.fftshift(np.fft.fft(np.array(ligne))))
        mag_freq = mag_freq.flatten()
        pks, _ = find_peaks(mag_freq, height=25)

        # si filtre
        if n_filter > 1:
            filter = self.smoothing_filter()
            freq_2 = filter.apply(freq_2)

        # faire correspondre index avec frequence FFT
        f_DOP = [
            (index - self.Nc / 2) / (self.Nc * self.te*n)
            for index in pks
        ]

        # equation vitesse avec frequence Doppler
        V = [
            c1*f_DOP/(2*self.fmin)
            for f_DOP in f_DOP
        ]
        if len(V) == 0:
            print("Erreur : Aucun pic détecté pour le calcul de la distance.(R)",V)
            return []
        print("la vitesse est de V:",V)
        return V
    
    def calculate_delta_phi(self, theta):
        """
        Calcule les différences de phase (delta_phi) pour chaque antenne en fonction de l'angle d'arrivée (theta).

        :param theta: Angle en degrés (-90° à 90°)
        :return: Liste des différences de phase (delta_phi) pour chaque antenne en radians
        """
        d = 0.00263  # Distance entre les antennes en mètres
        wavelength = 0.00526  # Longueur d'onde du signal radar en mètres
        theta_rad = math.radians(theta)  # Conversion en radians

        # Calcul des phases pour chaque antenne
        delta_phi_list = [
            (2 * math.pi * (n * d) * math.sin(theta_rad)) / wavelength
            for n in range(1, self.num_rx_antennas + 1)  # Boucle de 1 à num_rx_antennas
        ]
        
        return delta_phi_list

    def calc_AoA(self, phase_differences, d=0.00263, wavelength=0.00526):
        """
        Calcule l'angle d'arrivée (AoA) en utilisant les différences de phase entre les récepteurs.
        
        :param phase_differences: Liste des différences de phase entre les récepteurs
        :param d: Distance entre les antennes en mètres
        :param wavelength: Longueur d'onde du signal radar en mètres
        :return: Liste des angles d'arrivée des cibles
        """
        # Calcul de l'angle en utilisant la moyenne des différences de phase
        delta_phi = phase_differences[0]
        angle = np.arcsin((wavelength * delta_phi) / (2 * np.pi * d))
        angle_deg = np.degrees(angle)  # Conversion en degrés

        # Affichage Terminal
        print(f"Angle détecté (A) : {angle_deg:.2f}°")

        return angle_deg


    def FFT_Range_doppler_map(self, trimmed_data_np, count_Nc):

        trimmed_data_np = trimmed_data_np[0, :] # Prends uniquement la première antenne 

        # calcule La range FFT
        f = np.fft.fft(trimmed_data_np) 

        if self.smoothing_kernel_width > 1:
            filter = self.smoothing_filter(True)
            f = filter.apply(f)
        
        # Phase lissée avec cosinus
        phase = np.cos(np.unwrap(np.angle(f)))
        # prends que la partie de 0 a fe/2
        f = f [:int(150 / 2)]
        phase = phase[:int(150 / 2)]
        
        if count_Nc < self.Nc:
            if self.phase_matrix is None:
                self.phase_matrix = phase[np.newaxis, :]  # Initialiser avec une antenne
                self.fb_matrix = f[np.newaxis, :]
            else:
                # Vérification des dimensions avant vstack()
                if self.phase_matrix.shape[1] == phase.shape[0]:
                    self.phase_matrix = np.vstack((self.phase_matrix, phase))
                    self.fb_matrix = np.vstack((self.fb_matrix, f))
                else:
                    print(f"Dimension mismatch: phase_matrix {self.phase_matrix.shape} vs phase {phase.shape}")
            return None    

        if count_Nc==self.Nc:
            # matrice doppler
            fft_cols = np.fft.fft(self.phase_matrix, axis=0)
            # Prendre uniquement la moitié supérieure des fréquences (fréquences positives)
            fft_cols_positive = np.transpose(fft_cols[:fft_cols.shape[0] // 2, :])
            # matrice frequence battement
            fft_cols_fb=self.fb_matrix
            # Prendre uniquement les fréquences positives (si nécessaire)
            fft_cols_fb_positive = np.transpose(fft_cols_fb[:fft_cols_fb.shape[0] // 2, :])
            # multiplication des deux matrices
            mult=fft_cols_positive*fft_cols_fb_positive

            # Réinitialiser la matrice de phase
            self.phase_matrix = None  
            self.fb_matrix=None
            fft_cols=None

            return mult

    # Generation signal 
    def run(self):
        if self.server_addr is None:
            raw_data_gen = self.synthetic_data_gen(seq_len=150, chunk_len=200)


        # Deux passages dans la boucle
        #for i, raw_data in enumerate(raw_data_gen):
        #    print(f"Passage {i+1}: Données brutes (taille {len(raw_data)} octets) :", raw_data)
        #    if i == 4:  # Arrêter après 2 passages (i = 0, i = 1)
        #        break


        else:
            raw_data_gen = tcpclient.tcp_data_gen(self.server_addr, self.server_port,verbose=self.verbose)
        for seq, dir in self.iq_stream_converter.process(raw_data_gen):
            if not self.running:
                break
            #print(seq)
            self.ratemeter.notify([1, len(seq), 4 * len(seq)])

            with self.data_lock:
                # Convertir seq en tableau NumPy
                seq_np = np.array(seq, dtype=np.complex64)

                # Réarranger en prenant une valeur toutes les num_rx_antennas (pour les x antennes)
                self.data = np.array([seq_np[i::self.num_rx_antennas] for i in range(self.num_rx_antennas)])

                self.metadata = {"dir": dir}
                #print(self.data)
            self.update_signal.emit(self.data)
            # temps pour limiter signal synt
            time.sleep(0.05)
            

    def stop(self):
        self.running = False
        self.wait()

class PlotManager:
    def __init__(self, graphics_layout, data_thread,plot_widget, num_points=150, logscale=False, filter_PH=True):
        self.graphics_layout = graphics_layout
        self.data_thread = data_thread
        self.plot_widget=plot_widget
        self.num_points = num_points
        self.logscale = logscale
        self.filter_PH = filter_PH
        self.count_Nc=0

        # Crée des courbes pour les tracés I et Q
        self.i_curve = None
        self.q_curve = None
        self.markers_plot = None
        self.image=None

        # Type de graphique à afficher
        self.PLOT_RAW_SAMPLES = "raw_samples"
        self.PLOT_FFT_MAGNITUDE = "fft_magnitude"
        self.PLOT_RANGE = "range"
        self.PLOT_RANGE_DOPPLER_MAP = "Range_doppler_map"
        self.PLOT_ANGLE_OF_ARRIVAL= "Angle_of_Arrival"
        self.plot_type = self.PLOT_RAW_SAMPLES

    def high_pass_filter(self, data, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog = False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    # plot log
    def set_logscale(self, logscale):
        self.logscale = logscale
        print(f"PlotManager logscale set to: {self.logscale}")

    def set_filter_PH(self, filter_PH):
        self.filter_PH = filter_PH
        print(f"filter PH logscale set to: {self.filter_PH}")  

    
    # plot pour signal de base
    def plot_RAW(self, num_points=100, te=1 ,plot_title="", axeY=1):
        """Configure le graphique principal."""
        # periode d'echantillonage en ns
        te = te*1e6  # Facteur de conversion
        self.num_points = num_points

        # Ajouter la courbe
        self.i_curve = self.plot_widget.plot(pen='b', name="I-Data")
        self.plot_widget.setYRange(-axeY, axeY)           # Limites de l'axe Y
        self.plot_widget.setXRange(0, num_points)   # Limites de l'axe X
        self.plot_widget.setTitle(plot_title)
        self.plot_widget.showGrid(x=True, y=True)   # Affichage de la grille

        # Personnalisation de l'axe X
        axis = self.plot_widget.getAxis('bottom')   # Récupérer l'axe X
        axis.setLabel('Temps', units='us')          # Label pour l'axe X

        # Ajout d'une légende pour les antennes
        self.plot_widget.addLegend()

        # Génération dynamique des couleurs pour chaque antenne
        num_antennas = self.data_thread.num_rx_antennas
        antenna_colors = [pg.intColor(i, hues=num_antennas) for i in range(num_antennas)]

        # Création des courbes avec légendes
        self.i_curves = []
        for i in range(num_antennas):
            curve = self.plot_widget.plot(pen=antenna_colors[i], name=f"Antenna {i+1}")
            self.i_curves.append(curve)


        # Conversion des labels de l'axe X
        def x_axis_transform(value):
            """Transformation des ticks en fonction du facteur te."""
            return f"{value * te:.2f}"

        # Définir les ticks et leurs labels
        ticks = [(val, x_axis_transform(val)) for val in range(0, num_points + 1, 20)]
        axis.setTicks([ticks])

    # plot pour spectre de base
    def plot_FFT(self, num_points=100,te=1, plot_title="", maxY=20, minY=0):
        # periode d'echantillonage en ns
        fe = 1/(te*1e3)  # Facteur de conversion
        femin=fe/num_points
        
        self.plot_widget.setYRange(minY, maxY)             # Limites de l'axe Y
        self.plot_widget.setXRange(0,num_points/2)        # Limites de l'axe X
        self.plot_widget.setTitle(plot_title)
        self.i_curve = self.plot_widget.plot(pen='r', name="FFT-Magnitude")

        # Personnalisation de l'axe X
        axis = self.plot_widget.getAxis('bottom')  # Récupérer l'axe X
        axis.setLabel('distance', units='m')  # Label pour l'axe X

        # Conversion des labels de l'axe X
        def x_axis_transform(value):
            """Transformation des ticks en fonction du facteur femin."""
            return f"{value*3e8/(2*(self.data_thread.fmax-self.data_thread.fmin)) :.2f}"

        # Définir les ticks et leurs labels
        ticks = [(val, x_axis_transform(val)) for val in range(0, num_points + 1, 12)]
        axis.setTicks([ticks])

    # plot pour spectre de base
    def plot_FFT_log(self, num_points=100,te=1, plot_title="", maxY=20, minY=-20):
        # periode d'echantillonage en ns
        fe = 1/(te*1e3)  # Facteur de conversion
        femin=fe/num_points
        
        self.plot_widget.setYRange(minY, maxY)             # Limites de l'axe Y
        self.plot_widget.setXRange(0,num_points/2)        # Limites de l'axe X
        self.plot_widget.setTitle(plot_title)
        self.i_curve = self.plot_widget.plot(pen='r', name="FFT-Magnitude")

        # Personnalisation de l'axe X
        axis = self.plot_widget.getAxis('bottom')  # Récupérer l'axe X
        axis.setLabel('distance', units='m')  # Label pour l'axe X

        # Conversion des labels de l'axe X
        def x_axis_transform(value):
            """Transformation des ticks en fonction du facteur femin."""
            return f"{value*3e8/(2*(self.data_thread.fmax-self.data_thread.fmin)) :.2f}"

        # Définir les ticks et leurs labels
        ticks = [(val, x_axis_transform(val)) for val in range(0, num_points + 1, 12)]
        axis.setTicks([ticks])

    # plot pour visualisation distance
    def plot_1d_markers(self, x, xlim=None, title=None, xlabel=None, grid=None):
        """Affiche des marqueurs 1D dans un graphique."""

        # Configure les limites
        self.plot_widget.setXRange(xlim[0], xlim[1])
        self.plot_widget.setYRange(-1, 1)

        # Personnalisation de l'axe X
        axis = self.plot_widget.getAxis('bottom')   # Récupérer l'axe X
        axis.setLabel(xlabel, units='m')            # Label pour l'axe X

        # Définir les ticks de l'axe X de 0 à 20 avec un pas de 0.1
        ticks = [(i, f"{i:.1f}") for i in np.arange(0, 2.1, 0.1)]  # Ticks de 0 à 20 avec un pas de 0.1
        axis.setTicks([ticks])


        self.plot_widget.setTitle(title)

        
        # Si les marqueurs n'existent pas, créez-les
        self.markers_plot = self.plot_widget.plot(
            x,                  # Utilise self.x.detdata pour les positions des marqueurs
            np.zeros_like(x),   # Tous les marqueurs sont à y=0
            pen=None, 
            symbol='o', 
            symbolBrush='b'     
        )
        # Configure la grille si nécessaire
        if grid is not None and "axis" in grid:
            self.plot_widget.showGrid(
                x=True if grid["axis"] in ["x", "both"] else False,
                y=True if grid["axis"] in ["y", "both"] else False
            )
                
    # plot du Range doppler map
    def plot_range_doppler_map(self, trimmed_data_np,title=None):

        fs=1/self.data_thread.te
        n=fs*self.data_thread.Tc
        n=trimmed_data_np.shape[0]
        Nc=trimmed_data_np.shape[1]

        x_axis = 1e-4 * fs / n/1.1 * np.arange(0, n/1.1)  # Axe X
        y_axis = fs * self.data_thread.Tc / self.data_thread.Nc/1.5 * np.arange(0, self.data_thread.Nc//1.5)    # Axe Y
  
        # Ajouter un affichage d'image
        self.image = pg.ImageItem()
        self.plot_widget.addItem(self.image)

        # Appliquer une palette de couleurs (colormap)
        colormap = pg.colormap.get('plasma')  # Exemple de colormap : 'plasma', 'viridis', etc.
        self.image.setLookupTable(colormap.getLookupTable())

        # Afficher les données avec les axes personnalisés
        self.image.setImage(trimmed_data_np)

        vmin, vmax = np.min(trimmed_data_np), np.max(trimmed_data_np)
        self.image.setLevels([vmin, vmax])

        # Configuration des axes
        axis_x = self.plot_widget.getAxis('bottom')
        axis_y = self.plot_widget.getAxis('left')

        # Configurer les labels des axes
        axis_x.setLabel('distance', units='m')
        axis_y.setLabel('vitesse', units='m/s')


        # Conversion des labels de l'axe X
        def x_axis_transform(value):
            """Transformation des ticks en fonction du facteur fe."""
            return f"{value*3e8/(2*(self.data_thread.fmax-self.data_thread.fmin)) :.2f}"
        # Conversion des labels de l'axe Y
        def y_axis_transform(value):
            """Transformation des ticks en fonction du facteur fe."""
            return f"{value*1/(self.data_thread.Nc*self.data_thread.Tc)*3e8/(2*self.data_thread.fmin) :.2f}"


        # Configurer les ticks personnalisés pour l'axe X
        ticks_x = [(val, x_axis_transform(val)) for val in np.linspace(x_axis[0], x_axis[-1], num=17)]  # 17 ticks espacés
        axis_x.setTicks([ticks_x])

        # Configurer les ticks personnalisés pour l'axe Y
        ticks_y = [(val, y_axis_transform(val)) for val in np.linspace(y_axis[0], y_axis[-1], num=13)]  # 26 ticks espacés
        axis_y.setTicks([ticks_y])

        # Ajuster les plages des axes avec setRange
        self.plot_widget.setRange(xRange=(x_axis[0], x_axis[-1]), yRange=(y_axis[0], y_axis[-1]))

        # Ajouter une barre de couleur
        color_bar = pg.ColorBarItem(interactive=True, width=10, label="Amplitude")
        color_bar.setImageItem(self.image, insert_in=self.plot_widget)

        # Afficher la grille et lancer l'application
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.plot_widget.setTitle(title)

    def plot_AoA(self, num_points=180, plot_title="Angles", maxY=1.2, minY=-1.2):
        """Configure le graphique principal pour afficher les angles d’arrivée des cibles."""

        # Effacer les données précédentes
        self.plot_widget.clear()

        # Définir les limites de l’axe X et Y
        self.plot_widget.setYRange(minY, maxY)  
        self.plot_widget.setXRange(-90, 90)  # Plage des angles d’arrivée en degrés
        self.plot_widget.setTitle(plot_title)
        self.plot_widget.showGrid(x=True, y=True)  # Activation de la grille

        # Personnalisation de l'axe X
        axis = self.plot_widget.getAxis('bottom')
        axis.setLabel('Angle d’Arrivée', units='°')  # Label pour l’axe X

        # Définition des ticks de l’axe X (de -90° à 90° avec un pas de 15°)
        ticks = [(val, f"{val}°") for val in range(-90, 91, 15)]
        axis.setTicks([ticks])

        # Ajouter les points des cibles détectées
        self.markers_plot = self.plot_widget.plot(
            [], [],  # Initialisation avec des listes vides
            pen=None, 
            symbol='o', 
            symbolBrush='b'  # Marqueurs en bleu
        )

    # methode pour selection Plot
    def plot_data(self):
        """Trace les données en fonction du type de graphique."""
        with self.data_thread.data_lock:
            data = self.data_thread.data
            self.data_thread.data = None  # Efface les données après récupération

        if data is not None and len(data) > 0:
            data_np = np.array(data)

            if self.filter_PH:
                cutoff_frequency_hp = 50e3
                fe=1/(1e-6)

                data_np = self.high_pass_filter(data_np, cutoff_frequency_hp, fe)
                #data_np_avg = self.high_pass_filter(data_np_avg, cutoff_frequency_hp, fe)


            n = data_np.shape[1]
            x_axis = np.arange(n)

            if self.data_thread.window:
                if self.data_thread.window == "Rectangular":
                    window = 1
                elif self.data_thread.window == "Hanning":
                    window = np.hanning(n)
                elif self.data_thread.window == "Hamming":
                    window = np.hamming(n)
                elif self.data_thread.window == "Blackman":
                    window = np.blackman(n)
                else:
                    window = np.bartlett(n)
                
                data_np = data_np * window
                

            if self.plot_type == self.PLOT_RAW_SAMPLES:
                
                filter_instance = self.data_thread.smoothing_filter(True)  # Récupère le bon filtre

                # Vérifier le nombre d'antennes et appliquer le filtre correctement
                if data_np.ndim > 1:
                    data_np = np.apply_along_axis(lambda row: filter_instance.apply(row), axis=1, arr=data_np)
                else:
                    data_np = filter_instance.apply(data_np)


                # Extraire les valeurs réelles et imaginaires (I et Q)
                i_data = np.real(data_np)  # (4, 150)
                q_data = np.imag(data_np)  # (4, 150)
                
                # Vérifie si les courbes existent déjà, sinon les créer
                if self.i_curve is None:
                    self.i_curves = [self.plot_widget.plot(pen=pg.intColor(i + 4)) for i in range(i_data.shape[0])]     # Prends sur une seule antenne
                    self.plot_RAW(len(data_np[0]),self.data_thread.te,self.PLOT_RAW_SAMPLES,np.max(data_np[0].real))    # Prends sur une seule antenne


                if self.q_curve is None:
                    self.q_curves = [self.plot_widget.plot(pen=pg.intColor(i + 4)) for i in range(q_data.shape[0])]     # Prends sur une seule antenne

                num_antennas = self.data_thread.num_rx_antennas
                # Tracer les données I et Q pour chaque antenne
                for i in range(min(i_data.shape[0], num_antennas)):  # Boucle sur les x antennes
                    self.i_curves[i].setData(x_axis, i_data[i])  # Tracer I pour chaque antenne
                    self.q_curves[i].setData(x_axis, q_data[i])  # Tracer Q pour chaque antenne

            elif self.plot_type == self.PLOT_FFT_MAGNITUDE:
                
                #if self.smoothing_kernel_width > 1:
                #    filter = self.smoothing_filter(True)
                #    data_np = filter.apply(data_np)
                # FFT sur FIF
                f = np.fft.fft(data_np)

                if self.data_thread.smoothing_kernel_width > 1:
                    filter = self.data_thread.smoothing_filter(True)
                    f = filter.apply(f)
                
                angle=np.angle(f)
                frequence=np.abs(f[0]) # Prends sur une seule antenne 

                # si log
                if self.logscale:
                    f = 20 * np.log10(frequence + 1e-10)
                x_axis = np.linspace(0, len(frequence), len(frequence))
                # configuaration du plot si il n'existe pas
                if self.i_curve is None:
                    if not self.logscale:  
                        self.plot_FFT(len(frequence),self.data_thread.te,self.PLOT_FFT_MAGNITUDE,max(frequence))
                    else:
                        self.plot_FFT_log(len(frequence),self.data_thread.te,self.PLOT_FFT_MAGNITUDE,max(frequence),min(frequence))
                
                self.i_curve.setData(x_axis, frequence)

            elif self.plot_type == self.PLOT_RANGE:


                # Trace les distances
                R = self.data_thread.calc_range(data_np)
                # configuaration du plot si il n'existe pas
                if self.markers_plot is None:
                    self.plot_1d_markers(
                        R,
                        xlim=[0.1, 2],
                        title="Range",
                        xlabel="distance",
                        grid={"axis": "both"}
                    )
                self.markers_plot.setData(R, np.zeros_like(R))    

            elif self.plot_type == self.PLOT_RANGE_DOPPLER_MAP:
                # calcul et traitemnt de signal
                mult=self.data_thread.FFT_Range_doppler_map(data_np,self.count_Nc)
                # Chirp + 1
                self.count_Nc=self.count_Nc+1
                # si nombre Nc chirp est bon
                if mult is not None:
                    # Seuil pour binariser les données
                    threshold = 1000 # Ajustez cette valeur selon vos besoins
                    binary_data = np.where(mult > threshold, 1, 0)
                    # configuaration du plot si il n'existe pas
                    if self.image is None:
                        self.plot_range_doppler_map(binary_data,self.PLOT_RANGE_DOPPLER_MAP)

                    # Afficher les données avec les axes personnalisés
                    self.image.setImage(binary_data)
                    # V = self.data_thread.calc_Velocity(self.data_thread.phase_matrix, n_filter=self.smoothing_kernel_width)
                    self.count_Nc = 0

            elif self.plot_type == self.PLOT_ANGLE_OF_ARRIVAL:

                # Simulation d'une différence de phase entre récepteurs
                phase_differences = self.data_thread.calculate_delta_phi(self.data_thread.A)
                angle = self.data_thread.calc_AoA(phase_differences)  # Un seul angle retourné

                # Si le graphique n’est pas encore créé, on le configure
                if self.markers_plot is None:
                    self.plot_AoA()

                # Met à jour les données des marqueurs
                self.markers_plot.setData([float(angle)], [1])  # y = 1 pour tous les points




# class pour config du GUI (Slider et botton)
class GUI(QMainWindow):

    # titre des plots 
    PLOT_RAW_SAMPLES = "raw_samples"
    PLOT_FFT_MAGNITUDE = "fft_magnitude"
    PLOT_RANGE = "range"
    PLOT_RANGE_DOPPLER_MAP = "Range_doppler_map"
    PLOT_ANGLE_OF_ARRIVAL = "Angle_of_Arrival"

    def __init__(self, server_addr=None, server_port=None, verbose=False):
        super().__init__()

        #server adresse
        self.server_addr = server_addr
        self.server_port = server_port
        self.verbose = verbose
        self.logscale= False
        self.filter_PH= True

        # titre fenetre execution
        title = "RadarOC " + ("(synthetic data)" if server_addr is None else server_addr)
        self.setWindowTitle(title)
        self.resize(500, 500)
        self.plot_type = self.PLOT_RAW_SAMPLES  # Type de tracé initial

        # Initialisation du mode sombre / clair
        self.is_dark_mode = False

        # Configuration du layout principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(self)
        central_widget.setLayout(layout)


        # Création du widget GraphicsLayoutWidget pour PyQtGraph
        self.graphics_layout = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_layout)  # Ajouter le widget PyQtGraph au layout


        # Initialisation du thread de données
        self.data_thread = DataThread(server_addr=server_addr, server_port=server_port, verbose=verbose)
        self.data_thread.update_signal.connect(self.update_plot)
        self.data_thread.start()

        self.plot_widget = self.graphics_layout.addPlot()
       

        # Instancier PlotManager
        self.plot_manager = PlotManager(self.graphics_layout, self.data_thread,self.plot_widget,logscale=self.logscale, filter_PH=self.filter_PH)

        # Définir un type de graphique (par exemple, `PLOT_RAW_SAMPLES`)
        self.plot_manager.plot_type = self.plot_manager.PLOT_RAW_SAMPLES



        # Initialisation des widgets de contrôle
        self.init_config_panel(layout)


    def init_config_panel(self, layout):

        # bouton pour selection des plots
    

        self.tabs = QTabWidget()
        self.plot_tab_1=QWidget()
        plot_layout_1=QVBoxLayout(self.plot_tab_1)

        rd_frame = QGroupBox("Plot Type")
        rd_frame_layout=QVBoxLayout()
        rd_frame.setLayout(rd_frame_layout)

        plot_group = QButtonGroup(self)
        rd_raw = QRadioButton("Raw samples")
        rd_fft = QRadioButton("FFT magnitude")
        rd_range = QRadioButton("Range")
        rd_fft_anle = QRadioButton("Range_doppler_map")
        rd_angle = QRadioButton("Angle_of_Arrival")
        rd_raw.setChecked(True)

        # action d'appuyer sur boutton de Plot de selection
        rd_raw.toggled.connect(lambda: self.set_plot_type(self.PLOT_RAW_SAMPLES))
        rd_fft.toggled.connect(lambda: self.set_plot_type(self.PLOT_FFT_MAGNITUDE))
        rd_range.toggled.connect(lambda: self.set_plot_type(self.PLOT_RANGE))
        rd_fft_anle.toggled.connect(lambda: self.set_plot_type(self.PLOT_RANGE_DOPPLER_MAP))
        rd_angle.toggled.connect(lambda: self.set_plot_type(self.PLOT_ANGLE_OF_ARRIVAL))

        # Ajout des boutons au groupe et au layout
        plot_group.addButton(rd_raw)
        plot_group.addButton(rd_fft)
        plot_group.addButton(rd_range)
        plot_group.addButton(rd_fft_anle)
        plot_group.addButton(rd_angle)


        rd_frame_layout.addWidget(rd_raw)
        rd_frame_layout.addWidget(rd_fft)
        rd_frame_layout.addWidget(rd_range)
        rd_frame_layout.addWidget(rd_fft_anle)
        rd_frame_layout.addWidget(rd_angle)
        plot_layout_1.addWidget(rd_frame)

        # Checkbox pour l'échelle logarithmique
        self.logscale_checkbox = QCheckBox("Logarithmic scale")
        self.logscale_checkbox.setChecked(False)
        self.logscale_checkbox.setEnabled(False)
        self.logscale_checkbox.stateChanged.connect(self.toggle_logscale)
        self.logscale = self.logscale_checkbox.isChecked()

        plot_layout_1.addWidget(self.logscale_checkbox)

        self.plot_tab_1.setLayout(plot_layout_1)

        # Ajout de l'onglet au QTabWidget
        self.tabs.addTab(self.plot_tab_1, "Plot Type")

        # Ajout du QTabWidget au layout principal
        layout.addWidget(self.tabs)

        # Bouton pour basculer entre le mode sombre et clair
        self.toggle_theme_button = QPushButton("Mode Sombre")
        self.toggle_theme_button.clicked.connect(self.toggle_theme)

        # Créer un layout horizontal pour placer le bouton en bas à droite
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)  # Pousse le bouton vers la droite
        button_layout.addWidget(self.toggle_theme_button)  # Ajoute le bouton à droite

        # Ajouter le layout du bouton au layout principal (en bas)
        layout.addLayout(button_layout)


        if(self.server_addr == None):
            self.plot_tab_2=QWidget()
            plot_layout_2 = QVBoxLayout(self.plot_tab_2)
            # Création du slider pour modifier le parametre de la distance
            slider_frame = QGroupBox("Scale Factor")
            slider_frame_layout = QVBoxLayout()         # Ceci est défini ici, local à la méthode init_config_panel
            slider_frame.setLayout(slider_frame_layout)

            # Création du slider (de 0 à 10)
            self.scale_slider = QSlider(Qt.Horizontal)
            self.scale_slider.setRange(0, 10)           # Valeurs possibles entre 1 et 10
            self.scale_slider.setValue(3)               # Valeur par défaut
            self.scale_slider.setTickInterval(1)        # Intervalle de graduation
        
            # Création d'un label pour afficher la valeur actuelle du slider
            self.scale_label = QLabel("Scale Factor: 3")
            self.scale_slider.valueChanged.connect(self.update_scale_value)
        
            # Ajouter le slider et le label au layout
            slider_frame_layout.addWidget(self.scale_label)
            slider_frame_layout.addWidget(self.scale_slider)

            plot_layout_2.addWidget(slider_frame)
            self.plot_tab_2.setLayout(plot_layout_2)

            # Création du slider pour modifier le parametre de la vitesse 
            slider_velocity = QGroupBox("Velocity Factor")
            slider_velocity_layout = QVBoxLayout()          # Ceci est défini ici, local à la méthode init_config_panel
            slider_velocity.setLayout(slider_velocity_layout)

            # Création du slider (de -10 à 10)
            self.velocity_slider = QSlider(Qt.Horizontal)
            self.velocity_slider.setRange(-10, 10)  # Valeurs possibles entre 1 et 10
            self.velocity_slider.setValue(0)  # Valeur par défaut
            self.velocity_slider.setTickInterval(1)  # Intervalle de graduation
        
            # Création d'un label pour afficher la valeur actuelle du slider
            self.velocity_label = QLabel("velocity Factor: 0")
            self.velocity_slider.valueChanged.connect(self.update_velocity_value)
        
            # Ajouter le slider et le label au layout
            slider_velocity_layout.addWidget(self.velocity_label)
            slider_velocity_layout.addWidget(self.velocity_slider)
            plot_layout_2.addWidget(slider_velocity)

            self.plot_tab_2.setLayout(plot_layout_2)


            # Création du slider pour modifier le paramètre de l'angle
            slider_angle = QGroupBox("Angle Factor")
            slider_angle_layout = QVBoxLayout()
            slider_angle.setLayout(slider_angle_layout)

            # Création du slider (de -90 à 90)
            self.angle_slider = QSlider(Qt.Horizontal)
            self.angle_slider.setRange(-90, 90)  # Valeurs possibles entre 0 et 10
            self.angle_slider.setValue(15)  # Valeur par défaut
            self.angle_slider.setTickInterval(1)  # Intervalle de graduation

            # Création d'un label pour afficher la valeur actuelle du slider
            self.angle_label = QLabel("Angle Factor: 15")
            self.angle_slider.valueChanged.connect(self.update_angle_value)

            # Ajouter le slider et le label au layout
            slider_angle_layout.addWidget(self.angle_label)
            slider_angle_layout.addWidget(self.angle_slider)
            plot_layout_2.addWidget(slider_angle)

            self.plot_tab_2.setLayout(plot_layout_2)

            self.tabs.addTab(self.plot_tab_2, "Synthetique signal")

        self.plot_tab_3=QWidget()
        plot_layout_3 = QVBoxLayout(self.plot_tab_3)

        # Création d'une seule QGroupBox pour regrouper le tout
        smoothing_group = QGroupBox("Smoothing filter kernel width")
        smoothing_layout = QVBoxLayout()  # Layout principal de la box
        smoothing_group.setLayout(smoothing_layout)

        # Ajout du slider
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(1, 50)
        self.smoothing_slider.setValue(1)
        self.smoothing_slider.setTickInterval(1)

        self.smoothing_label = QLabel("1")
        self.smoothing_slider.valueChanged.connect(self.update_smoothing_value)

        smoothing_layout.addWidget(self.smoothing_label)
        smoothing_layout.addWidget(self.smoothing_slider)


        # Ajout des boutons IIR/FIR côte à côte
        button_layout = QHBoxLayout()  # Layout horizontal pour les boutons
        rd_IIR = QRadioButton("IIR")
        rd_FIR = QRadioButton("FIR")
        rd_FIR.setChecked(True)

        button_layout.addWidget(rd_IIR)
        button_layout.addWidget(rd_FIR)

        # Connecter les boutons à leurs actions
        rd_IIR.toggled.connect(lambda: self.set_smoothing_type(False))
        rd_FIR.toggled.connect(lambda: self.set_smoothing_type(True))

        # Ajouter les boutons au layout principal
        smoothing_layout.addLayout(button_layout)

        # Ajouter la QGroupBox globale au layout principal de la fenêtre
        plot_layout_3.addWidget(smoothing_group)
        self.plot_tab_3.setLayout(plot_layout_3)


        # Création du slider pour modifier le parametre de la distance
        slider_relative_peak_height = QGroupBox("relative peak height")
        slider_relative_peak_height_layout = QVBoxLayout()         # Ceci est défini ici, local à la méthode init_config_panel
        slider_relative_peak_height.setLayout(slider_relative_peak_height_layout)

        # Création du slider (de 0 à 10)
        self.relative_peak_height_slider = QSlider(Qt.Horizontal)
        self.relative_peak_height_slider.setRange(0, 40)           # Valeurs possibles entre 1 et 10
        self.relative_peak_height_slider.setValue(20)               # Valeur par défaut
        self.relative_peak_height_slider.setTickInterval(1)        # Intervalle de graduation
    
        # Création d'un label pour afficher la valeur actuelle du slider
        self.relative_peak_height_label = QLabel("relative_peak_height Factor: 20")
        self.relative_peak_height_slider.valueChanged.connect(self.update_peak_height_value)
    
        # Ajouter le slider et le label au layout
        slider_relative_peak_height_layout.addWidget(self.relative_peak_height_label)
        slider_relative_peak_height_layout.addWidget(self.relative_peak_height_slider)

        # Ajouter la QGroupBox globale au layout principal de la fenêtre
        plot_layout_3.addWidget(slider_relative_peak_height)
        self.plot_tab_3.setLayout(plot_layout_3)

        # Boutons radio pour la décimation
        rd_frame = QGroupBox("Decimation")
        rd_frame_layout = QHBoxLayout()  # Utilisation d'un layout horizontal
        rd_frame.setLayout(rd_frame_layout)

        decimation_group = QButtonGroup(rd_frame)
        decimation_options = [1, 10, 100, 1000]

        for value in decimation_options:
            # Création des boutons radio
            rd_button = QRadioButton(f"{value} ({'none' if value == 1 else ''})")
            rd_button.setChecked(value == 1)  # Activer le bouton "1" par défaut
            rd_button.toggled.connect(lambda checked, v=value: self.radio_decimation(v) if checked else None)

            # Ajouter le bouton au groupe et au layout horizontal
            decimation_group.addButton(rd_button)
            rd_frame_layout.addWidget(rd_button)

        # Ajouter la QGroupBox au layout principal
        plot_layout_3.addWidget(rd_frame)

        # Checkbox pour le filtre passe haut
        self.filter_PH_checkbox = QCheckBox("Filter PH")
        self.filter_PH_checkbox.setChecked(True)
        self.filter_PH_checkbox.setEnabled(True)
        self.filter_PH_checkbox.stateChanged.connect(self.toggle_filter_PH)
        self.filter_PH = self.filter_PH_checkbox.isChecked()

        plot_layout_3.addWidget(self.filter_PH_checkbox)

        self.plot_tab_3.setLayout(plot_layout_3)
        self.tabs.addTab(self.plot_tab_3, "Filter")

     # Boutons radio pour la décimation
        self.plot_tab_4=QWidget()
        plot_layout_4 = QVBoxLayout(self.plot_tab_4)

        rd_window = QGroupBox("Window")
        rd_window_layout = QHBoxLayout()  # Utilisation d'un layout horizontal
        rd_window.setLayout(rd_window_layout)

        window_group = QButtonGroup(rd_window)
        window_options = ["Rectangular","Hanning", "Hamming", "Blackman", "Triangular"]

        for value in window_options:
            # Création des boutons radio
            rd_button_window = QRadioButton(f"{value} ({'none' if value == 1 else ''})")
            rd_button_window.setChecked(value == "Rectangular")  # Activer le bouton "Rectangular" par défaut
            rd_button_window.toggled.connect(lambda checked, v=value: self.radio_window(v) if checked else None)

            # Ajouter le bouton au groupe et au layout horizontal
            window_group.addButton(rd_button_window)
            rd_window_layout.addWidget(rd_button_window)

        # Ajouter la QGroupBox au layout principal
        plot_layout_4.addWidget(rd_window)
        self.plot_tab_4.setLayout(plot_layout_4)
        self.tabs.addTab(self.plot_tab_4, "Windowing")

    # lie le slider distance avec la generation du signal
    def update_scale_value(self):
        # Mise à jour du label en fonction de la valeur du slider
        self.R = self.scale_slider.value()
        self.scale_label.setText(f"Scale Factor: {self.R}")
 
        # Mise à jour de R dans le thread de données
        self.data_thread.set_R(self.R)
    
        # Appeler une fonction pour mettre à jour le graphique ou les données avec ce facteur
        self.update_plot(self.R)

    # lie le slider vitesse avec la generation du signal
    def update_velocity_value(self):
        # Mise à jour du label en fonction de la valeur du slider
        self.V = self.velocity_slider.value()
        self.velocity_label.setText(f"Velocity Factor: {self.V}")

                
        # Mise à jour de V dans le thread de données
        self.data_thread.set_V(self.V)
    
        # Appeler une fonction pour mettre à jour le graphique ou les données avec ce facteur
        self.update_plot(self.V)


    def update_angle_value(self):
        # Mise à jour du label en fonction de la valeur du slider
        self.A = self.angle_slider.value()
        self.angle_label.setText(f"Angle Factor: {self.A}")

        # Mise à jour de V dans le thread de données
        self.data_thread.set_A(self.A)
    
        # Calcul de delta_phi pour chaque antenne
        delta_phi_list = self.data_thread.calculate_delta_phi(self.A)

        for i, delta_phi in enumerate(delta_phi_list, start=1):
            print(f"Antenna {i}: Δφ = {delta_phi:.4f} rad")

        # Appeler une fonction pour mettre à jour le graphique ou les données avec ce facteur
        self.update_plot(self.A)

    # lie le slider filtage avec la generation du signal
    def update_smoothing_value(self):
        # Mise à jour du label en fonction de la valeur du slider
        self.Smoothing = self.smoothing_slider.value()
        self.smoothing_label.setText(f"{self.Smoothing}")

        # Mise à jour de V dans le thread de données
        self.data_thread.set_Smoothing(self.Smoothing)
        
        # Appeler une fonction pour mettre à jour le graphique ou les données avec ce facteur
        self.update_plot(self.Smoothing)

    def set_smoothing_type(self, smoothing_type):

        self.data_thread.smoothing_fir = smoothing_type

    # lie le slider peak hight avec la generation du signal
    def update_peak_height_value(self):
        # Mise à jour du label en fonction de la valeur du slider
        self.peak_height = self.relative_peak_height_slider.value()
        self.relative_peak_height_label.setText(f"{self.peak_height}")

        # Mise à jour de V dans le thread de données
        self.data_thread.set_peak_height(self.peak_height)
    
        # Appeler une fonction pour mettre à jour le graphique ou les données avec ce facteur
        self.update_plot(self.peak_height)



    def radio_decimation(self, decimation):
        """Appelée lorsque l'un des boutons radio de décimation est sélectionné."""
        self.data_thread.set_decimation(decimation)     # Appelle la fonction de décimation avec la valeur sélectionnée
        print(f"Decimation set to: {decimation}")       # Débogage pour afficher la valeur sélectionnée

    def radio_window(self, Window):
        """Appelée lorsque l'un des boutons radio de décimation est sélectionné."""
        self.data_thread.set_window(Window)     # Appelle la fonction de décimation avec la valeur sélectionnée
        print(f"windowing set to: {Window}")       # Débogage pour afficher la valeur sélectionnée


    def set_plot_type(self, plot_type):
        # reinitialise tous les graphes
        self.plot_manager.plot_widget.clear()
        self.plot_widget.clear()
        self.plot_manager.q_curve=None
        self.plot_manager.i_curve=None
        self.plot_manager.markers_plot=None
        self.plot_manager.image=None
        # selectionne le plot voulu par l'utilisateur
        self.plot_type = plot_type
        self.plot_manager.plot_type = plot_type

        print(f"Plot type set to: {plot_type}")
    
        # Activation/désactivation de l'échelle logarithmique pour FFT uniquement
        is_fft = plot_type == self.PLOT_FFT_MAGNITUDE
        self.logscale_checkbox.setEnabled(is_fft)
        if not is_fft:
            self.logscale_checkbox.setChecked(False)

    def toggle_logscale(self):
        self.plot_manager.plot_widget.clear()
        self.plot_widget.clear()
        self.plot_manager.q_curve=None
        self.plot_manager.i_curve=None
        # Met à jour le drapeau d'échelle logarithmique en fonction de l'état de la case
        self.logscale = self.logscale_checkbox.isChecked()
        self.plot_manager.set_logscale(self.logscale)       # Met à jour PlotManager
        print(f"Logscale toggled to: {self.logscale}")      # Débogage
        self.update_plot(self.data_thread.data)             # Met à jour le graphique immédiatement

    def toggle_filter_PH(self):
        # Met à jour le drapeau d'échelle logarithmique en fonction de l'état de la case
        self.filter_PH = self.filter_PH_checkbox.isChecked()
        self.plot_manager.set_filter_PH(self.filter_PH)       # Met à jour PlotManager
        print(f"filter_PH toggled to: {self.filter_PH}")      # Débogage
        
    def toggle_theme(self):
        """Bascule entre le mode sombre et le mode clair"""
        if not hasattr(self, "is_dark_mode"):
            self.is_dark_mode = False  # Assure que l'attribut existe

        if not self.is_dark_mode:
            self.set_dark_mode()
        else:
            self.set_light_mode()
        self.is_dark_mode = not self.is_dark_mode

    def set_dark_mode(self):
        """Applique le mode sombre"""
        self.toggle_theme_button.setText("Mode Clair")
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        # Appliquer le thème sombre sur PyQtGraph
        pg.setConfigOption('background', 'k')  # Fond noir
        pg.setConfigOption('foreground', 'w')  # Texte blanc
        self.plot_widget.getViewBox().setBackgroundColor('k')
        self.tabs.setStyleSheet("QTabBar::tab { background: #333; color: white; } QTabBar::tab:selected { background: #555; }")
    def set_light_mode(self):
        """Applique le mode clair"""
        self.toggle_theme_button.setText("Mode Sombre")
        self.setStyleSheet("background-color: white; color: black;")

        # Appliquer le thème clair sur PyQtGraph
        pg.setConfigOption('background', 'w')  # Fond blanc
        pg.setConfigOption('foreground', 'k')  # Texte noir
        self.plot_widget.getViewBox().setBackgroundColor('w')
        self.tabs.setStyleSheet("QTabBar::tab { background: #f0f0f0; color: black; } QTabBar::tab:selected { background: #ddd; }")



    def update_plot(self, _):
        # Appelle `plot_data` pour afficher les nouvelles données
        self.plot_manager.plot_data()


    def closeEvent(self, event):
        self.data_thread.stop()
        event.accept()

# lance le prog
def run_gui(**kwargs):
    app = QApplication(sys.argv)
    gui = GUI(**kwargs)
    gui.show()
    sys.exit(app.exec_())

# prends en argument : "--server 192.168.0.10:9998"
def main(argv):
    server_addr, server_port, verbose = None, None, False
    import getopt

    try:
        options, args = getopt.getopt(argv[1:], "", [
            "help",
            "server=",
            "verbose",
        ])
        for name, value in options:
            if name == "--help":
                help()
                sys.exit(0)
            elif name == "--server":
                server_addr, server_port = value.split(":")
                server_port = int(server_port)
            elif name == "--verbose":
                verbose = True
    except Exception:
        print(traceback.format_exc())
        help(file=sys.stderr)
        sys.exit(1)

    run_gui(server_addr=server_addr, server_port=server_port, verbose=verbose)

if __name__ == "__main__":
    main(sys.argv)
