import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from threading import Thread
import glob
import os
import sys
import time
import pandas as pd
import numpy as np
import re
import datetime
from datetime import datetime
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkcalendar import Calendar
from datetime import datetime, timedelta
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy import crs as ccrs, feature as cfeature
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy.optimize import nnls
import csv
from scipy.stats import norm
from sklearn.linear_model import Lasso




logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')









# Define these at the module level
global_start_date = None
global_end_date = None
min_lat = None
min_lon = None
max_lat = None
max_lon = None








def RangeGeo(master):
    geo_window = tk.Toplevel(master)
    geo_window.title("Latitude and Longitude Limits")

    geo_frame = tk.LabelFrame(geo_window, text="Enter Coordinate Limits", padx=10, pady=10)
    geo_frame.pack(padx=10, pady=10)

    labels = ["Min Latitude:", "Max Latitude:", "Min Longitude:", "Max Longitude:"]
    entries = {}

    for i, label in enumerate(labels):
        tk.Label(geo_frame, text=label).grid(row=i, column=0, sticky="e", padx=5, pady=5)
        entry = tk.Entry(geo_frame, width=20)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[label] = entry

    result = {}

    def on_apply():
        try:
            result["min_lat"] = float(entries["Min Latitude:"].get())
            result["max_lat"] = float(entries["Max Latitude:"].get())
            result["min_lon"] = float(entries["Min Longitude:"].get())
            result["max_lon"] = float(entries["Max Longitude:"].get())
            geo_window.destroy()
        except ValueError:
            print("Please enter valid numeric values.")

    tk.Button(geo_window, text="Apply", command=on_apply).pack(pady=10)

    master.wait_window(geo_window)
    return result.get("min_lat"), result.get("max_lat"), result.get("min_lon"), result.get("max_lon")










########real library

def fwhm_energy_dep(E):
    # Your FWHM formula based on energy E (keV)
    return 0.5 + 0.05 * np.sqrt(E) + 0.001 * E



def photopeak(energy, peak_energy, area, fwhm_keV):
    sigma = fwhm_keV / 2.355
    return area * np.exp(-0.5 * ((energy - peak_energy) / sigma) ** 2)



def environmental_scatter(energy, intensity=0.1, decay_rate=0.002):
    scatter = np.exp(-decay_rate * energy)
    scatter /= np.max(scatter)  # Normalize shape
    return intensity * scatter


def compton_continuum_semiempirical(energy, peak_energy, area, beta=4.0, m=2.0):
    fwhm_keV = 0.5 + 0.05 * np.sqrt(peak_energy) + 0.001 * peak_energy
    m_e = 511.0  # Electron rest mass in keV
    E0 = peak_energy

    # Compton edge energy
    E_ce = E0 * (1 - 1 / (1 + (2 * E0 / m_e)))

    # Mask for valid Compton region
    valid = (energy > 0) & (energy < E_ce)
    E = energy[valid] / E_ce

    if np.sum(valid) == 0:
        return np.zeros_like(energy)

    # Semi-empirical function shape (normalized)
    shape = (E ** m) * np.exp(-beta * E)
    shape /= np.sum(shape)  # Normalize to unit area

    # Allocate area
    spectrum = np.zeros_like(energy)
    spectrum[valid] = shape * area

    # Optional Gaussian smoothing
    sigma = fwhm_keV / 2.355
    if sigma == 0 or (energy[-1] - energy[0]) == 0:
        return spectrum

    kernel_size = int(6 * sigma * len(energy) / (energy[-1] - energy[0]))
    if kernel_size < 3:
        return spectrum

    kernel_x = np.linspace(-3*sigma, 3*sigma, kernel_size)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= np.sum(kernel)

    if len(kernel) == 0 or len(spectrum) == 0:
        return spectrum

    # Apply smoothing via convolution
    smoothed = np.convolve(spectrum, kernel, mode='same')
    return smoothed




def skewed_gaussian(x, xi, omega, alpha):
    t = (x - xi) / omega
    return 2 / omega * norm.pdf(t) * norm.cdf(alpha * t)



def backscatter_peak(energy, peak_energy, area, skewness=5):
    fwhm_keV = 0.5 + 0.05 * np.sqrt(peak_energy) + 0.001 * peak_energy

    m_e = 511.0  # keV electron rest mass
    E_bs = peak_energy / (1 + 2 * peak_energy / m_e)
    sigma = (fwhm_keV * 3) / 2.355  
    peak = skewed_gaussian(energy, E_bs, sigma, skewness)
    peak *= area * 0.05 / peak.sum()
    return peak




def sigma_energy_dep(E):
    return fwhm_energy_dep(E) / 2.355

def pair_production_probability(E_p, alpha=0.001):
    if E_p < 1022:
        return 0
    return 1 - np.exp(-alpha * (E_p - 1022))

def improved_pair_production(energy, peak_energy, area, P_escape=0.1, include_double_escape=True):
    P_pair = pair_production_probability(peak_energy)
    if P_pair == 0:
        return np.zeros_like(energy)
    
    # Single escape peak: E = E_p - 511
    E_annihilation = peak_energy - 511
    area_pp = area * P_pair * P_escape
    fwhm_pp = fwhm_energy_dep(E_annihilation)
    peak = photopeak(energy, E_annihilation, area_pp, fwhm_pp)
    
    # Optional: double escape peak: E = E_p - 1022
    double_escape = np.zeros_like(energy)
    if include_double_escape:
        E_double = peak_energy - 1022
        area_dp = area * P_pair * P_escape**2
        fwhm_dp = fwhm_energy_dep(E_double)
        double_escape = photopeak(energy, E_double, area_dp, fwhm_dp)
    
    return peak + double_escape



def simulate_nai_response(peaks_keV, counts, channels=1024, e_min=0, e_max=3000):
    energy = np.linspace(e_min, e_max, channels)
    spectrum = np.zeros_like(energy)

    for peak, count in zip(peaks_keV, counts):
        fwhm_keV = fwhm_energy_dep(peak)
        spectrum += photopeak(energy, peak, count, fwhm_keV)
        spectrum += compton_continuum_semiempirical(energy, peak, count * 0.6)
        spectrum += backscatter_peak(energy, peak, count)
        spectrum += improved_pair_production(energy, peak, count, P_escape=0.1)

    spectrum += environmental_scatter(energy, intensity=0.2) * np.sum(spectrum) * 0.05
    return energy, spectrum








def generate_library_nai( default_count=1000):
    radioactive_materials = {
        'Cesium-137': [662],               # Common from nuclear fallout and ocean discharge
        'Cobalt-60': [1170, 1330],           # From reactor leaks, medical/industrial waste
        'Strontium-90': [546],             # Fission product, soluble in water
        'Tritium-3': [18.6],               # From nuclear reactors, also used in fusion
        'Iodine-131': [364],               # Short-lived but released in accidents
        'Plutonium-239': [414],            # Fallout, nuclear weapons testing
        'Plutonium-238': [43],            # Found near nuclear waste sites or accidents
        'Uranium-238': [63],              # Natural and anthropogenic sources
        'Uranium-235': [185],
        'Radium-226': [186],               # Naturally occurring, also from industrial waste
        'Radon-222': [352],                # Naturally decaying from radium in sediments
        'Carbon-14': [156],                # Naturally occurring, also from nuclear tests
        'Technetium-99m': [140],           # Medical waste discharge
        'Beryllium-7': [478],              # Cosmogenic, deposited from atmosphere into oceans
        'Americium-241': [59],            # Fallout, long-lived
        'Manganese-54': [835],             # From reactors and fallout
        'Zinc-65': [1115],                  # From nuclear and industrial discharges
        'Barium-133': [356, 81],        # Fission product
        'Lead-210': [46.5],                # Natural decay product, accumulates in sediments
        'Polonium-210': [51],             # Found in marine organisms (bioaccumulation)
        'Thorium-232': [63],              # Naturally occurring, particulate-bound in seawater
        'Neptunium-239': [30],            # Short-lived, fallout-related
        'Rubidium-86': [1078],              # Fission product
        'Silver-110m': [657],              # From nuclear reactor coolant leakage
        'Lanthanum-140': [1460],            # Fission product, high in fallout
        'Cesium-134': [796],               # Shorter-lived isotope of Cs, from reactor leaks (e.g., Fukushima)
        'Iodine-129': [39.6],              # Long-lived iodine isotope, traceable in ocean water from nuclear reprocessing
        'Ruthenium-106': [512],            # Fission product, found in marine fallout
        'Curium-244': [91],               # From nuclear waste; found in sediments near dump sites
        'Curium-242': [160],               # Shorter-lived curium isotope in some nuclear releases
        'Technetium-99': [140],            # Long-lived beta emitter, mobile in seawater, from reprocessing sites (Sellafield, La Hague)
        'Antimony-125': [176],             # Found in nuclear waste discharge to sea
        'Zirconium-95': [756],             # Fallout and reactor-related fission product
        'Niobium-95': [765],               # Accompanies Zr-95, another fission product in marine fallout
        'Chlorine-36' : [709],
        'Europium-152': [122, 244, 344],
        'Europium-154': [123, 247, 723],
        'Promethium-147': [122],
        'Tellurium-132': [228],
        'Cerium-144': [133],
        'Yttrium-90': [2.3],
        'Yttrium-91': [1554],
        'Neptunium-237': [312],
        'Thorium-228': [69, 238],
        'Thorium-230': [67],
        'Actinium-227': [75],
        'Protactinium-231': [312],
        'Francium-223': [81],
        'Bismuth-212': [727],
        'Bismuth-214': [609, 1120, 1764],
        'Lead-212': [238],
        'Lead-214': [352, 609],
        'Thallium-208': [583, 2615],
        'Krypton-85': [514],
        'Argon-41': [1293],
        'Scandium-46': [889, 1121],
        'Scandium-48': [1038, 1312],
        'Iron-59': [1099, 1292],
        'Cobalt-58': [810, 863],
        'Nickel-63': [67],
        'Nickel-59': [26],
        'Molybdenum-99': [739, 181],
        'Rhodium-105': [319],
        'Cadmium-109': [88],
        'Indium-111': [171, 245],
        'Tin-123': [159],
        'Samarium-151': [22],
        'Gadolinium-153': [103, 45],
        'Terbium-160': [879],
        'Hafnium-181': [133, 482],
        'Tantalum-182': [112, 1122],
        'Iridium-192': [316, 468],
        'Gold-198': [412],
        'Mercury-203': [279],
        'Cesium-136': [818],
        'Cesium-138': [662],
        'Iodine-132': [667],
        'Iodine-133': [529],
        'Iodine-135': [1130],
        'Tellurium-127m': [88],
        'Tellurium-129m': [105],
        'Tellurium-134': [846],
        'Xenon-133': [81],
        'Xenon-135': [249],
        'Xenon-137': [455],
        'Xenon-140': [537],
        'Rubidium-87': [275],
        'Rubidium-82': [776],
        'Bromine-82': [554],
        'Selenium-75': [136, 265],
        'Krypton-87': [511],
        'Krypton-88': [944],
        'Yttrium-92': [1017],
        'Zirconium-93': [67],
        'Ruthenium-103': [497],
        'Ruthenium-105': [318],
        'Rhodium-106': [511],
        'Palladium-107': [187],
        'Tin-121m': [240],
        'Tin-126': [415],
        'Tellurium-121m': [144],
        'Barium-140': [537],
        'Barium-141': [170],
        'Lanthanum-138': [1436],
        'Neodymium-147': [91],
        'Dysprosium-165': [94],
        'Erbium-169': [110],
        'Thulium-170': [84],
        'Ytterbium-175': [396],
        'Osmium-191': [129],
        'Rhenium-186': [137],
        'Rhenium-188': [155],
        'Lead-203': [279],
        'Bismuth-207': [570],
    }

    library = {}
    for name, peaks in radioactive_materials.items():
        counts = [default_count] * len(peaks)
        energies, spectrum = simulate_nai_response(peaks, counts)
        library[name] = (energies, spectrum)
    return library







class RangeCalendar:
    def __init__(self, master):
        # Use a Toplevel so it opens in its own window
        self.top = tk.Toplevel(master)
        self.top.title("Select Date Range")

        self.start_date = None
        self.end_date = None
        
        self.calendar = Calendar(self.top, selectmode='day', year=2025, month=6, day=12, font=("Helvetica", 16))
        self.calendar.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        self.calendar.bind("<<CalendarSelected>>", self.date_selected)
        
        self.label = tk.Label(self.top, text="Select start and end dates")
        self.label.grid(row=1, column=0, columnspan=2)

        # Add a Done button to close the window
        self.done_button = tk.Button(self.top, text="Done", command=self.close_window)
        self.done_button.grid(row=2, column=0, columnspan=2, pady=10)

    def close_window(self):
        self.top.destroy()

    def date_selected(self, event):
        global global_start_date, global_end_date  # use global variables

        selected = datetime.strptime(self.calendar.get_date(), "%m/%d/%y")
        
        if not self.start_date:
            self.start_date = selected
            self.end_date = None
            global_start_date = self.start_date
            global_end_date = None
            self.label.config(text=f"Start: {self.start_date.strftime('%Y-%m-%d')}")
            self.refresh_tags()
        elif not self.end_date:
            if selected < self.start_date:
                self.start_date, self.end_date = selected, self.start_date
            else:
                self.end_date = selected
            global_start_date = self.start_date
            global_end_date = self.end_date
            self.label.config(text=f"From {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            self.refresh_tags()
        else:
            self.start_date = selected
            self.end_date = None
            global_start_date = self.start_date
            global_end_date = None
            self.label.config(text=f"Start: {self.start_date.strftime('%Y-%m-%d')}")
            self.refresh_tags()

    def refresh_tags(self):
        self.calendar.calevent_remove('all')
        self.calendar.tag_config('range', background='lightblue')
        self.calendar.tag_config('start', background='skyblue')
        self.calendar.tag_config('end', background='deepskyblue')

        if self.start_date:
            self.calendar.calevent_create(self.start_date, 'Start', 'start')
        if self.end_date:
            self.calendar.calevent_create(self.end_date, 'End', 'end')
            current = self.start_date + timedelta(days=1)
            while current < self.end_date:
                self.calendar.calevent_create(current, 'In range', 'range')
                current += timedelta(days=1)







root = tk.Tk()
root.title("Demo - OGS Oceanography Radiation Monitoring System")
#screen_width = root.winfo_screenwidth()
#screen_height = root.winfo_screenheight()
#root.geometry(f"{screen_width}x{screen_height}")

top_frame = tk.Frame(root)
logo_path = '../OGS_Logo.jpg'

logo_image = Image.open(logo_path).resize((200, 200), Image.Resampling.LANCZOS)
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(top_frame, image=logo_photo)
logo_label.image = logo_photo
logo_label.grid(row=0, column=0, columnspan=2, pady=10)
top_frame.grid()  # This line is required!


# --- Top Frame ---
top_frame = tk.Frame(root)
top_frame.grid(padx=10, pady=10)

# --- First Fitting Method Selection Dropdown ---

# Create a container frame in row=1, column=1
fitting_container = tk.Frame(top_frame)
fitting_container.grid(row=1, column=1, padx=10, pady=10, sticky="n")

# Add all LabelFrames inside this container using grid
fitting_frame_1 = ttk.LabelFrame(fitting_container, text="Select your application")
fitting_frame_1.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

fitting_frame_2 = ttk.LabelFrame(fitting_container, text="Select radionuclide identifying method")
fitting_frame_2.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

fitting_frame_3 = ttk.LabelFrame(fitting_container, text="Select Background and noise lib")
fitting_frame_3.grid(row=2, column=0, padx=5, pady=5, sticky="ew")












fitting_methods = ["Oceangraphy", "general application"]
selected_method_1 = tk.StringVar(value=fitting_methods[0])

fitting_dropdown_1 = ttk.Combobox(
    fitting_frame_1,
    textvariable=selected_method_1,
    values=fitting_methods,
    state="readonly",
    width=30
)
fitting_dropdown_1.grid(padx=10, pady=10)





def confirm_selection_1():
    print(f"Selected method 1: {selected_method_1.get()}")

ttk.Button(fitting_frame_1, text="Confirm Selection", command=confirm_selection_1).grid(pady=5)

# --- Second Fitting Method Selection Dropdown ---

loc_methods = ["NNLS", "LASSO", "simple peaks"]
selected_method_2 = tk.StringVar(value=loc_methods[0])

fitting_dropdown_2 = ttk.Combobox(
    fitting_frame_2,
    textvariable=selected_method_2,
    values=loc_methods,
    state="readonly",
    width=30
)
fitting_dropdown_2.grid(padx=10, pady=5)

def confirm_selection_2():
    print(f"Selected method 2: {selected_method_2.get()}")

ttk.Button(fitting_frame_2, text="Confirm Selection", command=confirm_selection_2).grid(pady=5)



# --- Second Fitting Method Selection Dropdown -

loc_methods = ["Savitzky-Golay", "Ocean Background without auto noise remove", "general Background with auto noise remove", "general Background without auto noise remove", "non background nor noise remove"]
selected_method_3 = tk.StringVar(value=loc_methods[0])

fitting_dropdown_3 = ttk.Combobox(
    fitting_frame_3,
    textvariable=selected_method_3,
    values=loc_methods,
    state="readonly",
    width=30
)
fitting_dropdown_3.grid(padx=10, pady=5)

def confirm_selection_3():
    print(f"Selected method 3: {selected_method_3.get()}")

ttk.Button(fitting_frame_3, text="Confirm Selection", command=confirm_selection_3).grid(pady=5)







# Right panel for help label and console
right_panel = tk.Frame(root)
right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Help message label (safely packed inside right_panel)
help_label = tk.Label(right_panel, text="")
help_label.grid(pady=5)

# Scrolled Text output (bash-like window, also inside right_panel)
console_output = scrolledtext.ScrolledText(
    right_panel, width=70, height=17, wrap=tk.WORD, font=("Courier", 10)
)
console_output.grid()


# Right panel for help label and console
right_panel2 = tk.Frame(root)
right_panel2.grid(row=1, column=1   , padx=10, pady=10, sticky="n")


# Scrolled Text output (bash-like window, also inside right_panel)
console_output2 = scrolledtext.ScrolledText(
    right_panel2, width=70, height=17, wrap=tk.WORD, font=("Courier", 10)
)
console_output2.grid()





# --- Logo ---
#logo_path = '/home/asus/OGS-Projects/FirstProject-LuraBassi/Radiation monitoring system/n42 file convertor/OGS-Logo.jpg'
#logo_image = Image.open(logo_path).resize((150, 150), Image.Resampling.LANCZOS)
#logo_photo = ImageTk.PhotoImage(logo_image)
#logo_label = tk.Label(top_frame, image=logo_photo)
#logo_label.image = logo_photo  # Prevent garbage collection
#logo_label.grid(row=1, column=1, padx=100, pady=10)




# --- Spectrum Plot Frame 1 ---
spectrum_frame = ttk.LabelFrame(root, text="Spectrum Viewer 1")
spectrum_frame.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')

# Create first Matplotlib figure
fig = Figure(figsize=(5, 4), dpi=100)
spectrum_ax = fig.add_subplot(111)
spectrum_ax.set_title("Total Countrate from ... up ...", fontsize=10)
spectrum_ax.set_xlabel("Channel")
spectrum_ax.set_ylabel("Counts")

# Embed the first plot
canvas = FigureCanvasTkAgg(fig, master=spectrum_frame)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')


# --- Spectrum Plot Frame 2 ---
spectrum_frame2 = ttk.LabelFrame(root, text="Spectrum Viewer 2")
spectrum_frame2.grid(row=3, column=1, padx=10, pady=10, sticky='nsew')

# Create second Matplotlib figure
fig2 = Figure(figsize=(5, 4), dpi=100)
spectrum_ax2 = fig2.add_subplot(111)
spectrum_ax2.set_title("Total Countrate in same time in channel related to ... ", fontsize=10)
spectrum_ax2.set_xlabel("Channel")
spectrum_ax2.set_ylabel("Counts")

# Embed the second plot
canvas2 = FigureCanvasTkAgg(fig2, master=spectrum_frame2)
canvas2.draw()
canvas2.get_tk_widget().grid(row=0, column=0, sticky='nsew')





spectrum_frame3 = ttk.LabelFrame(root, text="Cartopy")
spectrum_frame3.grid(row=3, column=2, padx=10, pady=10, sticky='nsew')

# Create Matplotlib figure with Cartopy projection
fig3 = Figure(figsize=(7, 4), dpi=100)
spectrum_ax3 = fig3.add_subplot(111, projection=ccrs.PlateCarree())

# Optional: Add map features
spectrum_ax3.add_feature(cfeature.COASTLINE)
spectrum_ax3.add_feature(cfeature.BORDERS, linestyle=':')
spectrum_ax3.add_feature(cfeature.LAND, facecolor='lightgray')

# Initial title/labels (optional if plotting immediately after)
spectrum_ax3.set_title("Cartopy")
spectrum_ax3.set_xlabel("Longitude")
spectrum_ax3.set_ylabel("Latitude")

# Embed the plot in the Tkinter GUI
canvas3 = FigureCanvasTkAgg(fig3, master=spectrum_frame3)
canvas3.draw()
canvas3.get_tk_widget().grid(row=0, column=0, sticky='nsew')







spectrum_frame4 = ttk.LabelFrame(root, text="Cartopy")
spectrum_frame4.grid(row=0, column=2, rowspan=3, padx=10, pady=10, sticky='nsew')

# Create Matplotlib figure with Cartopy projection
fig4 = Figure(figsize=(7, 5), dpi=100)
spectrum_ax4 = fig4.add_subplot(111, projection=ccrs.PlateCarree())

# Optional: Add map features
spectrum_ax4.add_feature(cfeature.COASTLINE)
spectrum_ax4.add_feature(cfeature.BORDERS, linestyle=':')
spectrum_ax4.add_feature(cfeature.LAND, facecolor='lightgray')

# Initial title/labels (optional if plotting immediately after)
spectrum_ax4.set_title("Cartopy")
spectrum_ax4.set_xlabel("Longitude")
spectrum_ax4.set_ylabel("Latitude")

# Embed the plot in the Tkinter GUI
canvas4 = FigureCanvasTkAgg(fig4, master=spectrum_frame4)
canvas4.draw()
canvas4.get_tk_widget().grid(row=0, column=0, sticky='nsew')






def load_folder():
    folder_path = filedialog.askdirectory(title="Select a folder containing RS250*.n42 files and RS250*.csv file.")
    if folder_path:
        append_to_console(f"Selected folder: {folder_path}")
        
        def periodic_processing():
            while True:
                append_to_console("Processing folder...")
                process_data(folder_path)
                append_to_console("Waiting 600 seconds for next cycle...")

                for remaining in range(600, 0, -1):
                    append_to_console(f"Countdown: {remaining} seconds remaining")
                    time.sleep(1)
        
        # Start the periodic processing in a background thread
        thread = Thread(target=periodic_processing, daemon=True)
        thread.start()
    else:
        append_to_console("No folder selected.")



def show_help():
    help_label.config(text="Hi, I will be load!")

def append_to_console(text):
    console_output.insert(tk.END, text + '\n')
    console_output.see(tk.END)
   
def export_kml2(dose_data, output_path, Coordination_accumulation, fit_results=None):
    totaldose_accumulation_each_point = np.sum(dose_data, axis=1)
    
    lat = Coordination_accumulation[:, 0]
    lon = Coordination_accumulation[:, 1]
    val = totaldose_accumulation_each_point

    min_val = np.min(val)
    max_val = np.max(val)
    val_normalized = (val - min_val) / (max_val - min_val)

    def get_kml_color(value_norm):
        r = int(255 * value_norm)
        g = int(255 * (1 - value_norm))
        b = 0
        a = 255  
        return f'{a:02x}{b:02x}{g:02x}{r:02x}'

    with open(output_path, 'w') as fidKML:
        fidKML.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fidKML.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        fidKML.write('<Document>\n')
        fidKML.write('<Name>Dose Distribution Points</Name>\n')

        if fit_results is not None:
            isotope_names = list(fit_results.keys())
            num_points = len(Coordination_accumulation)

            for i in range(num_points):
                # Extract all isotope values at this point
                isotope_values = {iso: fit_results[iso][i] for iso in isotope_names}
                
                # Filter to get only significant ones (e.g., > 0.01)
                detected_isotopes = {iso: val for iso, val in isotope_values.items() if val > 0.01}
                if not detected_isotopes:
                    continue  # Skip if no isotope detected

                # Normalize to percentage
                total = sum(detected_isotopes.values())
                isotope_percentages = {
                    iso: 100 * val / total for iso, val in detected_isotopes.items()
                }

                # Format string with percentages
                radionuclides_str = ', '.join(
                    f"{iso} ({perc:.1f}%)" for iso, perc in isotope_percentages.items()
                )

                lat_i, lon_i = Coordination_accumulation[i]

                fidKML.write('<Placemark>\n')
                fidKML.write(f'<description><![CDATA[<p>This point estimated anomaly radionuclides are <b>{radionuclides_str}</b>.</p>]]></description>\n')
                fidKML.write('<Style>\n')
                fidKML.write('<IconStyle>\n')
                fidKML.write('<color>ff000000</color>\n')
                fidKML.write('<scale>1.5</scale>\n')
                fidKML.write('<Icon>\n')
                fidKML.write('<href>http://maps.google.com/mapfiles/kml/paddle/blk-flag.png</href>\n')
                fidKML.write('</Icon>\n')
                fidKML.write('</IconStyle>\n')
                fidKML.write('</Style>\n')
                fidKML.write('<Point>\n')
                fidKML.write(f'<coordinates>{lon_i:.6f},{lat_i:.6f},{100000:.2f}</coordinates>\n')
                fidKML.write('</Point>\n')
                fidKML.write('</Placemark>\n')

        for i in range(len(lon)):
            if val_normalized[i] > 0.06:
                color = get_kml_color(1)
                fidKML.write('<Placemark>\n')
                fidKML.write('<Style>\n')
                fidKML.write('<IconStyle>\n')
                fidKML.write(f'<color>{color}</color>\n')
                fidKML.write('<scale>1.5</scale>\n')
                fidKML.write('<Icon>\n')
                fidKML.write('<href>http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png</href>\n')
                fidKML.write('</Icon>\n')
                fidKML.write('</IconStyle>\n')
                fidKML.write('</Style>\n')
                fidKML.write('<Point>\n')
                fidKML.write(f'<coordinates>{lon[i]:.6f},{lat[i]:.6f},{val[i]:.2f}</coordinates>\n')
                fidKML.write('</Point>\n')
                fidKML.write('</Placemark>\n')

        for i in range(len(lon)):
            color = get_kml_color(0.01)
            fidKML.write('<Placemark>\n')
            fidKML.write('<Style>\n')
            fidKML.write('<IconStyle>\n')
            fidKML.write(f'<color>{color}</color>\n')
            fidKML.write('<scale>0.4</scale>\n')
            fidKML.write('<Icon>\n')
            fidKML.write('<href>http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png</href>\n')
            fidKML.write('</Icon>\n')
            fidKML.write('</IconStyle>\n')
            fidKML.write('</Style>\n')
            fidKML.write('<Point>\n')
            fidKML.write(f'<coordinates>{lon[i]:.6f},{lat[i]:.6f},{val[i]:.2f}</coordinates>\n')
            fidKML.write('</Point>\n')
            fidKML.write('</Placemark>\n')

        fidKML.write('</Document>\n')
        fidKML.write('</kml>\n')
     
def find_simple_peaks(Dose_accumulation, anomali_coordination, anomali_energy, totaldose_accumulation_each_point,Coordination_accumulation):
 




    #peaks, properties = find_peaks(totaldose_accumulation_each_point, height=5)
    # peaks, properties = anomali_energy , anomali_coordination
    peak_coords = Coordination_accumulation
    peaks = 609 * np.ones(len(peak_coords))
 #   append_to_console(f'{peak_coords}')
  #  peaks, properties = anomali_energy , anomali_coordination
    return peaks, peak_coords





def find_peaks_Lasso(Dose_accumulation, Coordination_accumulation, alpha=0.001):
    lib = generate_library_nai()
    isotope_names = list(lib.keys())

    A = np.column_stack([lib[name][1] for name in isotope_names])
    fit_results = {iso: np.zeros(Dose_accumulation.shape[0]) for iso in isotope_names}

    for i in range(Dose_accumulation.shape[0]):
        spectrum = Dose_accumulation[i]

        model = Lasso(alpha=alpha, positive=True, max_iter=10000)
        model.fit(A, spectrum)
        coeffs = model.coef_

        for iso, coeff in zip(isotope_names, coeffs):
            fit_results[iso][i] = coeff

    # CSV writing same as before...
    return fit_results, Coordination_accumulation







def find_peaks_nnls(Dose_accumulation, Coordination_accumulation):
    # Generate library (assuming this function exists)
    lib = generate_library_nai()
    isotope_names = list(lib.keys())

    # Build matrix A with shape (1024, number_of_isotopes)
    A = np.column_stack([lib[name][1] for name in isotope_names])

    # Initialize result dictionary
    fit_results = {iso: np.zeros(Dose_accumulation.shape[0]) for iso in isotope_names}

    # Fit each spectrum (row) individually
    for i in range(Dose_accumulation.shape[0]):
        spectrum = Dose_accumulation[i]  # Removed random noise addition
        coeffs, _ = nnls(A, spectrum)
        for iso, coeff in zip(isotope_names, coeffs):
            fit_results[iso][i] = coeff

    # Write results to CSV with proper header
# Write results to CSV with proper header
    with open("fit_result.csv", "w", newline='') as f:
        writer = csv.writer(f)

        # Construct header dynamically
        header = ["Spectrum Index"] + list(fit_results.keys()) + ["Coordination"]
        writer.writerow(header)

        num_spectra = Dose_accumulation.shape[0]
        for i in range(num_spectra):
            row = [i] + [fit_results[iso][i] for iso in fit_results.keys()] + [Coordination_accumulation[i]]
            writer.writerow(row)


    return fit_results, Coordination_accumulation











###################################################







def process_n42_data(folder_path, data, real_time_accumulation, latitud, longitud,  global_start_date, global_end_date,air_temp, pressure, rel_humidity):
    append_to_console(f"Processing N42 files from {folder_path}...")
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.n42')]
    Dose_accumulation = []
    Coordination_accumulation_fake = []
    Fake_time_accumulation = []

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        append_to_console(f'Processing file: {file_path}')

        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                data = f.read()

            target_line = '<ChannelData Compression="CountedZeroes">'
            target_line2 = '          </ChannelData>'
            coordination = '<Coordinates>'
            coordination2 = '</Coordinates>'
            time = '<StartTime>'
            time2 = '</StartTime>'

            start_idx = data.find(target_line) + len(target_line)
            end_idx = data.find(target_line2)
            start_idx_coordination = data.find(coordination) + len(coordination)
            end_idx_coordination = data.find(coordination2)
            start_idx_time = data.find(time) + len(time)
            end_idx_time = data.find(time2)

            if start_idx != -1 and end_idx != -1:
                data_segment = data[start_idx:end_idx].strip()
                data_segment = re.sub(r'\s+', ' ', data_segment)
                numbers = np.fromstring(data_segment, sep=' ')
                matrix = []
                i = 0
                while i < len(numbers):
                    if numbers[i] == 0:
                        matrix.extend([0] * int(numbers[i + 1]))
                        i += 2
                    else:
                        matrix.append(numbers[i])
                        i += 1
                Dose_accumulation.append(matrix)

            if start_idx_coordination != -1 and end_idx_coordination != -1:
                data_segment = data[start_idx_coordination:end_idx_coordination].strip()
                data_segment = re.sub(r'\s+', ' ', data_segment)
                numbers = np.fromstring(data_segment, sep=' ')
                matrix = list(numbers)
                Coordination_accumulation_fake.append(matrix)

            if start_idx_time != -1 and end_idx_time != -1:
                data_segment = data[start_idx_time:end_idx_time].strip()
                data_segment = re.sub(r'\s+', ' ', data_segment)
                dt = datetime.strptime(data_segment, '%Y-%m-%dT%H:%M:%SZ')
                Fake_time_accumulation.append(dt)

        else:
            append_to_console(f'File does not exist: {file_path}')

    
    
    

    
    Coordination_accumulation_fake = np.array(Coordination_accumulation_fake)

    Fake_time_accumulation = np.array(Fake_time_accumulation)
    Dose_accumulation = np.array(Dose_accumulation)

    #fake time 31570
    # Sort by Fake_time_accumulation and apply the same order to related arrays
    sort_idx = np.argsort(Fake_time_accumulation)

    # Apply sorting
    Fake_time_accumulation = Fake_time_accumulation[sort_idx]

 

    Dose_accumulation = Dose_accumulation[sort_idx]




    Coordination_accumulation_fake = Coordination_accumulation_fake[sort_idx]
        
    Coordination_accumulation =[]

    column_0 = Coordination_accumulation_fake[:, 0]
    column_1 = Coordination_accumulation_fake[:, 1]
    
 


    # 1) find insertion positions
    idxs = np.searchsorted(real_time_accumulation, Fake_time_accumulation)
    append_to_console('1111111)')

    # 2) mask out-of-bounds before indexing
    in_bounds = idxs < real_time_accumulation.size
    safe_idxs = np.minimum(idxs, real_time_accumulation.size - 1)

    # 3) compare only in-bound entries
    valid = in_bounds & (
        real_time_accumulation[safe_idxs] == Fake_time_accumulation
    )
    append_to_console(f"valid count: {valid.sum()}")

    # 4) allocate & assign

    column_0[valid] = latitud[safe_idxs[valid]]
    column_1[valid] = longitud[safe_idxs[valid]]




    Coordination_accumulation_fake_1 = column_0.flatten()
    Coordination_accumulation_fake_2 =  column_1.flatten()
    Coordination_accumulation_fake_11 = Coordination_accumulation_fake_1.reshape(-1, 1)
    Coordination_accumulation_fake_12 = Coordination_accumulation_fake_2.reshape(-1, 1)

    Coordination_accumulation = np.hstack([Coordination_accumulation_fake_11, Coordination_accumulation_fake_12])
    


    append_to_console("done!")







 #   for k in range(Coordination_accumulation_index):
 #       for i in range(longitud_index):
 #           if Fake_time_accumulation[k] == real_time_accumulation[i]:
  #              column_0[k] = latitud[i]
 #               column_1[k] = longitud[i]


    append_to_console("File processing complete.")









    # Further processing (e.g., anomaly detection, distance calculation, plotting)
    calculate_dose(folder_path, Dose_accumulation, Coordination_accumulation, Fake_time_accumulation, real_time_accumulation, global_start_date, global_end_date,air_temp, pressure, rel_humidity,min_lat, max_lat, min_lon, max_lon)
    # Example: plot the first spectrum as a test















def process_data(folder_path):
    append_to_console("Processing files...")
    
    csv_files = glob.glob(os.path.join(folder_path, 'rs250_win*.csv'))
    
    if not csv_files:
        append_to_console("No matching CSV files found.")
    else:
        data = pd.read_csv(csv_files[0])
        append_to_console("CSV file read successfully.")
     #   append_to_console(f'{data}')
        latitud = data['latitude']
        longitud = data['longitude']
        air_temp = data['air_temp']
        pressure = data['pressure']
        rel_humidity = data['rel_humidity']
        real_time = data['time'].astype(str)
        real_time_accumulation = pd.to_datetime(real_time, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        process_n42_data(folder_path, data, real_time_accumulation, latitud, longitud,  global_start_date, global_end_date,air_temp, pressure, rel_humidity)

    append_to_console('Im after csv')


 # Use the correct function name here


def Savitzky_Golay(Dose_accumulation, window_length=11, polyorder=3):
    """
    Denoise all spectra in Dose_accumulation using Savitzky-Golay filter.

    Parameters:
    - Dose_accumulation: numpy array (N_spectra, N_channels)
    - window_length: int, odd number, smoothing window length
    - polyorder: int, polynomial order for filter

    Returns:
    - denoised_spectra: numpy array same shape as Dose_accumulation
    """
    # Validate window_length (must be odd and <= number of channels)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > Dose_accumulation.shape[1]:
        window_length = Dose_accumulation.shape[1] - (1 - Dose_accumulation.shape[1] % 2)

    denoised_spectra = savgol_filter(Dose_accumulation,
                                     window_length=window_length,
                                     polyorder=polyorder,
                                     axis=1)
    Dose_accumulation = denoised_spectra
    return Dose_accumulation






def subtraction_Oceanbckg_nonoiseremove(Dose_accumulation):
    Initial_Dose_accumulation = Dose_accumulation
    noise_accumulation = 3 * np.ones_like(Dose_accumulation)
    Dose_accumulation = -3 * Initial_Dose_accumulation
    append_to_console("2")
    return Dose_accumulation

def subtraction_Generalbckg_noiseremove(Dose_accumulation):
    Initial_Dose_accumulation = Dose_accumulation
    noise_accumulation = 3 * np.ones_like(Dose_accumulation)
    Dose_accumulation = -3 * Initial_Dose_accumulation
    append_to_console("3")
    return Dose_accumulation

def subtraction_Generalbckg_nonoiseremove(Dose_accumulation):
    Initial_Dose_accumulation = Dose_accumulation
    noise_accumulation = 3 * np.ones_like(Dose_accumulation)
    Dose_accumulation = -3 * Initial_Dose_accumulation
    append_to_console("4")
    return Dose_accumulation

def subtraction_nobckg_nonoiseremove(Dose_accumulation):
    append_to_console("5")
    return Dose_accumulation


def calculate_dose(folder_path, Dose_accumulation, Coordination_accumulation,Fake_time_accumulation,real_time_accumulation , global_start_date, global_end_date,air_temp, pressure, rel_humidity,min_lat, max_lat, min_lon, max_lon):
    append_to_console("Calculating dose...")




    
    append_to_console(f'ggggggg{min_lat}, {max_lat}, {min_lon}, {max_lon}')


    if selected_method_3.get() == "Savitzky-Golay":
        Dose_accumulation = Savitzky_Golay(Dose_accumulation)    

    if selected_method_3.get() == "Ocean Background without auto noise remove":
        Dose_accumulation = subtraction_Oceanbckg_nonoiseremove(Dose_accumulation)    

    if selected_method_3.get() == "general Background with auto noise remove":
        Dose_accumulation = subtraction_Generalbckg_noiseremove(Dose_accumulation)    

    if selected_method_3.get() == "general Background without auto noise remove":
        Dose_accumulation = subtraction_Generalbckg_nonoiseremove(Dose_accumulation)

    if selected_method_3.get() == "non background nor noise remove":
        Dose_accumulation = subtraction_nobckg_nonoiseremove(Dose_accumulation)


    totaldose_accumulation_each_point = np.sum(Dose_accumulation, axis=1)
    #totaldose 31000 coordinationacumulation31000 doseacumulation31000
    #realtimeaccumulation 18000

    #append_to_console(f'erererererer{real_time_accumulation}')
 # Check if time range is valid before using it
    use_time_selection = global_start_date is not None and global_end_date is not None

    if use_time_selection:
        try:
            start_idx_new = np.searchsorted(Fake_time_accumulation, global_start_date, side='left')
            end_idx_new = np.searchsorted(Fake_time_accumulation, global_end_date, side='right') - 1
        except Exception as e:
            append_to_console(f"Error in time-based index selection: {e}")
            return
    else:
        start_idx_new = end_idx_new = None  # fallback to coordinate selection
        append_to_console("Time selection is not provided or invalid. Falling back to coordinate selection.")

    # If time-based selection not used, try coordinate-based selection
    if start_idx_new is None or end_idx_new is None:
        if Coordination_accumulation is None or Coordination_accumulation.size == 0:
            append_to_console("Error: Coordination data is missing or empty.")
            return

        if None in (min_lat, max_lat, min_lon, max_lon):
            append_to_console("Error: Latitude or Longitude bounds are None.")
            return

        try:
            mask = (
                (Coordination_accumulation[:, 0] >= float(min_lat)) & (Coordination_accumulation[:, 0] <= float(max_lat)) &
                (Coordination_accumulation[:, 1] >= float(min_lon)) & (Coordination_accumulation[:, 1] <= float(max_lon))
            )
            append_to_console(f"Valid coordinate filter applied. Match count: {np.sum(mask)}")

            matching_indices = np.where(mask)[0]

            if matching_indices.size > 0:
                start_idx_new = matching_indices[0]
                end_idx_new = matching_indices[-1]
            else:
                start_idx_new = end_idx_new = -1
                append_to_console("Warning: No coordinates matched the selected bounds.")
        except Exception as e:
            append_to_console(f"Error while filtering coordinates: {e}")
            return




















    x = Fake_time_accumulation[start_idx_new:end_idx_new]
    y = totaldose_accumulation_each_point[start_idx_new:end_idx_new]
    z = Dose_accumulation[start_idx_new:end_idx_new, 203]


    if selected_method_4.get() == "Bi_214 Energy Channel":
        seeking_channel = 203
    elif selected_method_4.get() == "Cs_137 Energy Channel":
        seeking_channel = 221    
    elif selected_method_4.get() == "Be_7 Energy Channel":
        seeking_channel = 160

 





    spectrum_ax.clear()

    spectrum_ax.scatter(x, y, color='blue', s=10)

    if global_start_date is not None and global_end_date is not None:
        spectrum_ax.set_title(f"Total Countrate from {global_start_date} up to {global_end_date}", fontsize=10)
    elif None not in (max_lat, min_lat, max_lon, min_lon):
        spectrum_ax.set_title(f"Total Countrate from {min_lon},{min_lat} up to {max_lon},{max_lat}", fontsize=10)



    spectrum_ax.set_xlabel("Time")
    spectrum_ax.set_ylabel("Counts")
    canvas.draw() 


    spectrum_ax2.clear()

    spectrum_ax2.scatter(x, z, color='red', s=10)
    spectrum_ax2.set_title(f"Total Countrate in same time in channel related to {selected_method_4.get()}", fontsize=10)
    spectrum_ax2.set_xlabel("Time")
    spectrum_ax2.set_ylabel("Counts")
    canvas2.draw() 



    dd = totaldose_accumulation_each_point.reshape(-1, 1)
    yy = Coordination_accumulation[:, 1]  # Longitude
    zz = Coordination_accumulation[:, 0]  # Latitude
    append_to_console(f"Dose shape: {dd.shape}, Coord shape: {Coordination_accumulation.shape}")



    fig3.clf()  # clear the figure completely
    ax = fig3.add_subplot(111, projection=ccrs.PlateCarree())
    
    # Add map features again
    ax.set_global()
    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    ax.add_feature(cfeature.RIVERS, edgecolor='cornflowerblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    ax.add_feature(cfeature.LAKES, edgecolor='navy', facecolor='aliceblue')

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


    # Plot the new data
    sc = ax.scatter(yy, zz, c = dd, cmap='RdYlGn_r', s=0.002, transform=ccrs.PlateCarree())
    # Create a colorbar axes on the right side of the figure
    cbar_ax = fig4.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig4.colorbar(sc, cax=cbar_ax, orientation='vertical')
    cbar.set_label("Do", fontsize=10)

    # Labels and title
    ax.set_title("All World Radiation Map, Hitmap, and Contour line", fontsize=12, weight='bold')
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)

    canvas3.draw()


    # Clear the figure and add high-res PlateCarree projection
    fig4.clf()
    ax = fig4.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude = 180))
    ee = totaldose_accumulation_each_point[start_idx_new:end_idx_new].reshape(-1, 1)
    yyy = Coordination_accumulation[start_idx_new:end_idx_new, 1]  # Longitude
    yyy = np.where(yyy < 0 , yyy + 360 , yyy)
    zzz = Coordination_accumulation[start_idx_new:end_idx_new, 0]


    append_to_console(f'a{yyy}')
    append_to_console(f'cc{zzz}')

    valid_mask = (yyy > 0) & (yyy < 360) & (zzz > -90) & (zzz < 90)
    append_to_console(f'b{valid_mask}')

    # Apply mask to all three arrays
    dose_filtered = ee[valid_mask]
    lon_filtered = yyy[valid_mask]
    lat_filtered = zzz[valid_mask]

    append_to_console(f'ffff{lat_filtered},{lon_filtered},{dose_filtered}')
    append_to_console('dfdfdfdfccwq')





    lon_grid = np.linspace(np.min(lon_filtered), np.max(lon_filtered), 100).reshape(-1, 1)
    append_to_console(f'hhh{lon_grid}')

    lat_grid = np.linspace(np.min(lat_filtered), np.max(lat_filtered), 100).reshape(-1, 1)
    append_to_console(f'ggg{lat_grid}')

    lat_grid = np.linspace(np.min(lat_filtered), np.max(lat_filtered), 100).reshape(-1,1)
    lon_grid = np.linspace(np.min(lon_filtered), np.max(lon_filtered), 100).reshape(-1,1)

    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid, indexing='ij')  # shapes (100, 100)

    pairs = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))  # shape (10000, 2)

    append_to_console(f'sssssfdfdfccwq{lat_grid},{lon_grid}')
    lon_valid = []
    lat_valid = []

    for i in range(len(pairs)):
        for j in range(len(lon_filtered)):
            distance = abs(pairs[i,1] - lon_filtered[j]) + abs(pairs[i,0] - lat_filtered[j])
            if distance < 2:
                append_to_console(f'{i}')
                append_to_console(f'{pairs[i,1]}')
                lon_valid.append(pairs[i, 1])  # lon
                lat_valid.append(pairs[i, 0])  # lat


                

    append_to_console('fbfbfgvdfvdtf')



    # Step 3 - Interpolate   nearest
    try:
        dose_grid = griddata(
            (lon_filtered, lat_filtered),
            dose_filtered,
            (lon_valid, lat_valid),
            method='cubic'
        )
    except Exception as e:
        append_to_console(f"Griddata error: {str(e)}")
        return
    



    lon_filtered = yyy[(yyy > 0) & (yyy < 360)]
    lat_filtered = zzz[(zzz > -90) & (zzz < 90)]

    if lon_filtered.size > 0 and lat_filtered.size > 0:
        ax.set_extent([
            max(np.min(lon_filtered) - 10, -180),
            min(np.max(lon_filtered) + 10, 180),
            max(np.min(lat_filtered) - 10, -90),
            min(np.max(lat_filtered) + 10, 90)
        ], crs=ccrs.PlateCarree())




    # Compute extent from data with a small margin




    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    ax.add_feature(cfeature.RIVERS, edgecolor='cornflowerblue')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    ax.add_feature(cfeature.LAKES, edgecolor='navy', facecolor='aliceblue')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False


    # Extract data


    # Plot data with improved styling
    sc = ax.scatter(yyy, zzz, c = ee, cmap='bwr', s=0.1, transform=ccrs.PlateCarree())
    sc = ax.scatter(lon_valid, lat_valid, c = dose_grid, cmap='RdYlGn_r', s=0.1, transform=ccrs.PlateCarree())


    # Add colorbar
    cbar_ax = fig4.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig4.colorbar(sc, cax=cbar_ax, orientation='vertical')
    cbar.set_label("Do", fontsize=10)

    # Labels and title


    if global_start_date is not None and global_end_date is not None:
        ax.set_title(f"Radiation Map During Laura Bassi Vessel Cruise {global_start_date} up to {global_end_date} ", fontsize=8, weight='bold')
    elif None not in (max_lat, min_lat, max_lon, min_lon):
        ax.set_title(f"Radiation Map During Laura Bassi Vessel Cruise {min_lon},{min_lat} up to {max_lon},{max_lat} ", fontsize=8, weight='bold')

    #ax.set_title(f"Radiation Map During Laura Bassi Vessel Cruise {global_start_date} up to {global_end_date} ", fontsize=8, weight='bold')
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)

    canvas4.draw()




























    dz = np.diff(z)
    dy = np.diff(y)

    # Compute correlation between the variations
    corr_coef, p_value = pearsonr(dz, dy)
    console_output2.insert(tk.END, f"The corrolation between Bi-214 energy channel and total countrate variations is {corr_coef} and P-value is {p_value} \n ")
    console_output2.insert(tk.END, "-------------------------------------------------\n")
    console_output2.see(tk.END)
    
    zz = Dose_accumulation[start_idx_new:end_idx_new, 221]

    dzz = np.diff(zz)
    dy = np.diff(y)

    # Compute correlation between the variations
    corr_coef, p_value = pearsonr(dzz, dy)
    console_output2.insert(tk.END, f"The corrolation between Cs-137 energy channel and total countrate variations is {corr_coef} and P-value is {p_value} \n ")
    console_output2.insert(tk.END, "-------------------------------------------------\n")
    console_output2.see(tk.END)  
    


    zzz = Dose_accumulation[start_idx_new:end_idx_new, 160]

    dzzz = np.diff(zzz)
    dy = np.diff(y)
     # Compute correlation between the variations
    corr_coef, p_value = pearsonr(dzzz, dy)
    console_output2.insert(tk.END, f"The corrolation between Berelium-7 energy channel and total countrate variations is {corr_coef} and P-value is {p_value} \n ")
    console_output2.insert(tk.END, "-------------------------------------------------\n")
    console_output2.see(tk.END)    
    
    
    
    max_correlation = 0
    min_p_value = 1

    for i in range(1, 1023):  # i from 1 to 1024 inclusive
        k = Dose_accumulation[start_idx_new:end_idx_new, i]
        dk = np.diff(k)
        corr_coef, p_value = pearsonr(dk, dy)
        if corr_coef > max_correlation and p_value < min_p_value:
            max_correlation = corr_coef
            min_p_value = p_value
            max_correlation_energy = i * 3

    console_output2.insert(tk.END, f"The maximum correlation between total count rate and energy channels variations during {global_start_date} up to {global_end_date} is at {max_correlation_energy} keV, with correlation {max_correlation} and P-value is {min_p_value}.\n")
    console_output2.insert(tk.END, "-------------------------------------------------\n")
    console_output2.see(tk.END)


    #############################################################

    append_to_console(f'temp: {air_temp}\n')
    append_to_console(f'temp shape: {np.shape(air_temp)}\n')

    air_temp_accumulation = []

    for i in range(start_idx_new, end_idx_new):
        matched = False
        for j in range(len(real_time_accumulation)):
            if Fake_time_accumulation[i] == real_time_accumulation[j]:
                append_to_console(f'++++++++++++\n')
                temp = air_temp[j]
                matched = True
                break

        if not matched:
            if i == 0:
                temp = air_temp[0]  # Handle start
            else:
                temp = air_temp_accumulation[i - 1]  # Reuse previous value

        append_to_console(f'temp: {i}, {temp} \n')
        air_temp_accumulation.append(temp)

    append_to_console(f'temp: {air_temp_accumulation}\n')
    append_to_console(f'temp shape: {np.shape(air_temp_accumulation)}\n')

    # Prepare and validate arrays for correlation
    zdz = air_temp_accumulation[0:(end_idx_new - start_idx_new)]
    dzdz = np.diff(zdz)
    dy = np.array(dy, dtype=np.float64)
    dzdz = np.array(dzdz, dtype=np.float64)

    mask = np.isfinite(dzdz) & np.isfinite(dy)
    if np.std(dzdz[mask]) == 0 or np.std(dy[mask]) == 0:
        corr_coef, p_value = np.nan, np.nan
    else:
        corr_coef, p_value = pearsonr(dzdz[mask], dy[mask])

    console_output2.insert(tk.END, f"The correlation between air_temperature and total countrate variations is {corr_coef} and P-value is {p_value} \n")
    console_output2.insert(tk.END, "-------------------------------------------------\n")
    console_output2.see(tk.END)



    #########################################################


    append_to_console(f'pressure: {pressure}\n')
    append_to_console(f'pressure shape: {np.shape(pressure)}\n')

    pressure_accumulation = []

    for i in range(start_idx_new, end_idx_new):
        matched = False
        for j in range(len(real_time_accumulation)):
            if Fake_time_accumulation[i] == real_time_accumulation[j]:
                append_to_console(f'++++++++++++\n')
                press = pressure[j]
                matched = True
                break

        if not matched:
            if i == 0:
                press = pressure[0]  # Handle start
            else:
                press = pressure_accumulation[i - 1]  # Reuse previous value

        append_to_console(f'press: {i}, {press} \n')
        pressure_accumulation.append(press)


    append_to_console(f'press: {pressure_accumulation}\n')
    append_to_console(f'press shape: {np.shape(pressure_accumulation)}\n')

    # Prepare and validate arrays for correlation
    ddz = pressure_accumulation[0:(end_idx_new - start_idx_new)]
    dzdzd = np.diff(ddz)
    dy = np.array(dy, dtype=np.float64)
    dzdzd = np.array(dzdzd, dtype=np.float64)

    mask = np.isfinite(dzdzd) & np.isfinite(dy)
    if np.std(dzdzd[mask]) == 0 or np.std(dy[mask]) == 0:
        corr_coef, p_value = np.nan, np.nan
    else:
        corr_coef, p_value = pearsonr(dzdzd[mask], dy[mask])

    console_output2.insert(tk.END, f"The correlation between pressure and total countrate variations is {corr_coef} and P-value is {p_value} \n")
    console_output2.insert(tk.END, "-------------------------------------------------\n")
    console_output2.see(tk.END)



 ##########################################################################

    append_to_console(f'rel_humidity: {rel_humidity}\n')
    append_to_console(f'rel_humidity shape: {np.shape(rel_humidity)}\n')

    rel_humidity_accumulation = []

    for i in range(start_idx_new, end_idx_new):
        matched = False
        for j in range(len(real_time_accumulation)):
            if Fake_time_accumulation[i] == real_time_accumulation[j]:
                append_to_console(f'++++++++++++\n')
                hum = rel_humidity[j]
                matched = True
                break

        if not matched:
            if i == 0:
                hum = rel_humidity[0]  # Handle start
            else:
                hum = rel_humidity[i - 1]  # Reuse previous value

        append_to_console(f'humidity: {i}, {press} \n')
        rel_humidity_accumulation.append(hum)


    append_to_console(f'humidity: {rel_humidity_accumulation}\n')
    append_to_console(f'humidity shape: {np.shape(rel_humidity_accumulation)}\n')

    # Prepare and validate arrays for correlation
    ddzz = rel_humidity_accumulation[0:(end_idx_new - start_idx_new)]
    dzdzdm = np.diff(ddzz)
    dy = np.array(dy, dtype=np.float64)
    dzdzdm = np.array(dzdzdm, dtype=np.float64)

    mask = np.isfinite(dzdzdm) & np.isfinite(dy)
    if np.std(dzdzdm[mask]) == 0 or np.std(dy[mask]) == 0:
        corr_coef, p_value = np.nan, np.nan
    else:
        corr_coef, p_value = pearsonr(dzdzdm[mask], dy[mask])

    console_output2.insert(tk.END, f"The correlation between rel_humidity and total countrate variations is {corr_coef} and P-value is {p_value} \n")
    console_output2.insert(tk.END, "-------------------------------------------------\n")
    console_output2.see(tk.END)










































    totaldose_accumulation_incoordination = np.zeros(1024)
    
    for i in range(len(Dose_accumulation)):
        totaldose_accumulation_incoordination += Dose_accumulation[i]
    
    append_to_console("Dose calculation completed.")
    append_to_console(f"Calculated doses: {totaldose_accumulation_incoordination[:5]} ...") 
    
    Ocean_typic_Bckg = Dose_accumulation[np.argmin(Dose_accumulation[:][211])]


    Dose_accumulation = np.array(Dose_accumulation)
    Relation_DosetoBckg = np.ones_like(Dose_accumulation)
    anomali_coordination = []
    anomali_energy = []



    #####HPC needed part

    Dose_accumulation_shape_0 = Dose_accumulation.shape[0]
    Dose_accumulation_shape_1 = Dose_accumulation.shape[1]
    Dose_accumulation_40 =[]

    for i in range(Dose_accumulation_shape_0):  # iterate backwards to safely delete
        if Dose_accumulation[i, 40] < 40:
            del Dose_accumulation[i]
        else:
            Dose_accumulation_40.append(Dose_accumulation[i, 40])

    append_to_console('aaaaa')
    for j in range(len(Ocean_typic_Bckg)):
        if   Ocean_typic_Bckg[j] == 0:
             Ocean_typic_Bckg[j] = 1
    append_to_console('bbbbbb')

    Ocean_typic_Bckg_40 = Ocean_typic_Bckg[40]

    for i in range(Dose_accumulation_shape_0):
        for j in range(Dose_accumulation_shape_1):
            Relation_DosetoBckg[i, j] = (Ocean_typic_Bckg_40 * Dose_accumulation[i, j]) / (Dose_accumulation_40[i] * Ocean_typic_Bckg[j])

    append_to_console('cccccc')


    index = []

    for i in range(Dose_accumulation_shape_0):
        for j in range(Dose_accumulation_shape_1):
            if Relation_DosetoBckg[i, j] > 9:
                index.append((i, j))

                    
            elif Dose_accumulation[i, j] > 50 and Relation_DosetoBckg[i, j] > 4:
                index.append((i, j))


                    
            elif Dose_accumulation[i, j] > 500 and Relation_DosetoBckg[i, j] > 1.8:
                index.append((i, j))



    append_to_console('dddddd')



    for i, j in index:
        anomali_coordination.append(Coordination_accumulation[i])
        anomali_energy.append(j)
        
    append_to_console('eeee')



                

                    

 ###########HPC needed part


  #  append_to_console(f'Anomalous coordinates: {anomali_coordination},{anomali_energy}')
 


    if selected_method_2.get() == "simple peaks":
        peaks, properties = find_simple_peaks(Dose_accumulation, anomali_coordination,anomali_energy,totaldose_accumulation_each_point,Coordination_accumulation)
     #   append_to_console(f'a: {selected_method_1.get()}, {selected_method_2.get()}, {selected_method_3.get()}')

    if selected_method_2.get() == "LASSO":
        fit_results, properties = find_peaks_Lasso(Dose_accumulation, Coordination_accumulation, alpha=0.001)

    #    append_to_console(f'a: {selected_method_1.get()}, {selected_method_2.get()}, {selected_method_3.get()}')
    if selected_method_2.get() == "NNLS":
        fit_results, properties = find_peaks_nnls(Dose_accumulation,Coordination_accumulation)
     #   append_to_console(f'a: {selected_method_1.get()}, {selected_method_2.get()}, {selected_method_3.get()}')
   













    kml_output_path = os.path.join(folder_path, 'dose_distribution_points.kml')
    kml_output_path2 = os.path.join(folder_path, 'material_distribution_points.kml')

    anomalous_indices = list(set([coord[0] for coord in anomali_coordination]))
    
    export_kml2(Dose_accumulation, kml_output_path2, Coordination_accumulation,fit_results)
    
    append_to_console(f'KML file saved to: {kml_output_path}')
    return anomali_coordination




def OceanGraphyApplication(peaks, properties):
    radioactive_materials = {
        'Cesium-137': [662],               # Common from nuclear fallout and ocean discharge
        'Cobalt-60': [1170, 1330],           # From reactor leaks, medical/industrial waste
        'Strontium-90': [546],             # Fission product, soluble in water
        'Tritium-3': [18.6],               # From nuclear reactors, also used in fusion
        'Iodine-131': [364],               # Short-lived but released in accidents
        'Plutonium-239': [414],            # Fallout, nuclear weapons testing
        'Plutonium-238': [43],            # Found near nuclear waste sites or accidents
        'Uranium-238': [63],              # Natural and anthropogenic sources
        'Uranium-235': [185],
        'Radium-226': [186],               # Naturally occurring, also from industrial waste
        'Radon-222': [352],                # Naturally decaying from radium in sediments
        'Carbon-14': [156],                # Naturally occurring, also from nuclear tests
        'Technetium-99m': [140],           # Medical waste discharge
        'Beryllium-7': [478],              # Cosmogenic, deposited from atmosphere into oceans
        'Americium-241': [59],            # Fallout, long-lived
        'Manganese-54': [835],             # From reactors and fallout
        'Zinc-65': [1115],                  # From nuclear and industrial discharges
        'Barium-133': [356, 81],        # Fission product
        'Lead-210': [46.5],                # Natural decay product, accumulates in sediments
        'Polonium-210': [51],             # Found in marine organisms (bioaccumulation)
        'Thorium-232': [63],              # Naturally occurring, particulate-bound in seawater
        'Neptunium-239': [30],            # Short-lived, fallout-related
        'Rubidium-86': [1078],              # Fission product
        'Silver-110m': [657],              # From nuclear reactor coolant leakage
        'Lanthanum-140': [1460],            # Fission product, high in fallout
        'Cesium-134': [796],               # Shorter-lived isotope of Cs, from reactor leaks (e.g., Fukushima)
        'Iodine-129': [39.6],              # Long-lived iodine isotope, traceable in ocean water from nuclear reprocessing
        'Ruthenium-106': [512],            # Fission product, found in marine fallout
        'Curium-244': [91],               # From nuclear waste; found in sediments near dump sites
        'Curium-242': [160],               # Shorter-lived curium isotope in some nuclear releases
        'Technetium-99': [140],            # Long-lived beta emitter, mobile in seawater, from reprocessing sites (Sellafield, La Hague)
        'Antimony-125': [176],             # Found in nuclear waste discharge to sea
        'Zirconium-95': [756],             # Fallout and reactor-related fission product
        'Niobium-95': [765],               # Accompanies Zr-95, another fission product in marine fallout
        'Chlorine-36' : [709],
        'Bismuth-214' : [609],
    }

    Detected_radionuclide_list = []
    Detected_radionuclide_list_index = []
    Detected_radionuclide_list1 = []
    radioactive_materials_index =[]


    properties_len = len(properties)


    for i in range(properties_len):
        for isotope, energies in radioactive_materials.items():
            for energy in energies:
                if abs(peaks[i] - energy) < 5:
                    Detected_radionuclide_list_index.append(i)
                    radioactive_materials_index.append(isotope)


                    
    append_to_console(f'IIIIIIIIIIIIIIIIIIII{len(Detected_radionuclide_list_index)}')

    for k in range(len(Detected_radionuclide_list_index)):
        Detected_radionuclide_list.append((radioactive_materials_index[k],peaks[Detected_radionuclide_list_index[k]],properties[Detected_radionuclide_list_index[k]]))

    
    append_to_console('yes')

    return(Detected_radionuclide_list)







def GeneralApplication(peaks, properties):
    radioactive_materials = {
        'Cesium-137': [662],               # Common from nuclear fallout and ocean discharge
        'Cobalt-60': [1170, 1330],           # From reactor leaks, medical/industrial waste
        'Strontium-90': [546],             # Fission product, soluble in water
        'Tritium-3': [18.6],               # From nuclear reactors, also used in fusion
        'Iodine-131': [364],               # Short-lived but released in accidents
        'Plutonium-239': [414],            # Fallout, nuclear weapons testing
        'Plutonium-238': [43],            # Found near nuclear waste sites or accidents
        'Uranium-238': [63],              # Natural and anthropogenic sources
        'Uranium-235': [185],
        'Radium-226': [186],               # Naturally occurring, also from industrial waste
        'Radon-222': [352],                # Naturally decaying from radium in sediments
        'Carbon-14': [156],                # Naturally occurring, also from nuclear tests
        'Technetium-99m': [140],           # Medical waste discharge
        'Beryllium-7': [478],              # Cosmogenic, deposited from atmosphere into oceans
        'Americium-241': [59],            # Fallout, long-lived
        'Manganese-54': [835],             # From reactors and fallout
        'Zinc-65': [1115],                  # From nuclear and industrial discharges
        'Barium-133': [356, 81],        # Fission product
        'Lead-210': [46.5],                # Natural decay product, accumulates in sediments
        'Polonium-210': [51],             # Found in marine organisms (bioaccumulation)
        'Thorium-232': [63],              # Naturally occurring, particulate-bound in seawater
        'Neptunium-239': [30],            # Short-lived, fallout-related
        'Rubidium-86': [1078],              # Fission product
        'Silver-110m': [657],              # From nuclear reactor coolant leakage
        'Lanthanum-140': [1460],            # Fission product, high in fallout
        'Cesium-134': [796],               # Shorter-lived isotope of Cs, from reactor leaks (e.g., Fukushima)
        'Iodine-129': [39.6],              # Long-lived iodine isotope, traceable in ocean water from nuclear reprocessing
        'Ruthenium-106': [512],            # Fission product, found in marine fallout
        'Curium-244': [91],               # From nuclear waste; found in sediments near dump sites
        'Curium-242': [160],               # Shorter-lived curium isotope in some nuclear releases
        'Technetium-99': [140],            # Long-lived beta emitter, mobile in seawater, from reprocessing sites (Sellafield, La Hague)
        'Antimony-125': [176],             # Found in nuclear waste discharge to sea
        'Zirconium-95': [756],             # Fallout and reactor-related fission product
        'Niobium-95': [765],               # Accompanies Zr-95, another fission product in marine fallout
        'Chlorine-36' : [709],
        'Europium-152': [122, 244, 344],
        'Europium-154': [123, 247, 723],
        'Promethium-147': [122],
        'Tellurium-132': [228],
        'Cerium-144': [133],
        'Yttrium-90': [2.3],
        'Yttrium-91': [1554],
        'Neptunium-237': [312],
        'Thorium-228': [69, 238],
        'Thorium-230': [67],
        'Actinium-227': [75],
        'Protactinium-231': [312],
        'Francium-223': [81],
        'Bismuth-212': [727],
        'Bismuth-214': [609, 1120, 1764],
        'Lead-212': [238],
        'Lead-214': [352, 609],
        'Thallium-208': [583, 2615],
        'Krypton-85': [514],
        'Argon-41': [1293],
        'Scandium-46': [889, 1121],
        'Scandium-48': [1038, 1312],
        'Iron-59': [1099, 1292],
        'Cobalt-58': [810, 863],
        'Nickel-63': [67],
        'Nickel-59': [26],
        'Molybdenum-99': [739, 181],
        'Rhodium-105': [319],
        'Cadmium-109': [88],
        'Indium-111': [171, 245],
        'Tin-123': [159],
        'Samarium-151': [22],
        'Gadolinium-153': [103, 45],
        'Terbium-160': [879],
        'Hafnium-181': [133, 482],
        'Tantalum-182': [112, 1122],
        'Iridium-192': [316, 468],
        'Gold-198': [412],
        'Mercury-203': [279],
        'Cesium-136': [818],
        'Cesium-138': [662],
        'Iodine-132': [667],
        'Iodine-133': [529],
        'Iodine-135': [1130],
        'Tellurium-127m': [88],
        'Tellurium-129m': [105],
        'Tellurium-134': [846],
        'Xenon-133': [81],
        'Xenon-135': [249],
        'Xenon-137': [455],
        'Xenon-140': [537],
        'Rubidium-87': [275],
        'Rubidium-82': [776],
        'Bromine-82': [554],
        'Selenium-75': [136, 265],
        'Krypton-87': [511],
        'Krypton-88': [944],
        'Yttrium-92': [1017],
        'Zirconium-93': [67],
        'Ruthenium-103': [497],
        'Ruthenium-105': [318],
        'Rhodium-106': [511],
        'Palladium-107': [187],
        'Tin-121m': [240],
        'Tin-126': [415],
        'Tellurium-121m': [144],
        'Barium-140': [537],
        'Barium-141': [170],
        'Lanthanum-138': [1436],
        'Neodymium-147': [91],
        'Dysprosium-165': [94],
        'Erbium-169': [110],
        'Thulium-170': [84],
        'Ytterbium-175': [396],
        'Osmium-191': [129],
        'Rhenium-186': [137],
        'Rhenium-188': [155],
        'Lead-203': [279],
        'Bismuth-207': [570],


    }

    Detected_radionuclide_list = []
    Detected_radionuclide_list_index = []
    Detected_radionuclide_list1 = []
    radioactive_materials_index =[]


    properties_len = len(properties)


    for i in range(properties_len):
        for isotope, energies in radioactive_materials.items():
            for energy in energies:
                if abs(peaks[i] - energy) < 5:
                    Detected_radionuclide_list_index.append(i)
                    radioactive_materials_index.append(isotope)


                    
    append_to_console(f'IIIIIIIIIIIIIIIIIIII{len(Detected_radionuclide_list_index)}')

    for k in range(len(Detected_radionuclide_list_index)):
        Detected_radionuclide_list.append((radioactive_materials_index[k],peaks[Detected_radionuclide_list_index[k]],properties[Detected_radionuclide_list_index[k]]))

    
    append_to_console('yes')

    return(Detected_radionuclide_list)





def open_range_calendar():
    range_cal = RangeCalendar(root)


def open_geo_window():
    global min_lat, max_lat, min_lon, max_lon
    min_lat, max_lat, min_lon, max_lon = RangeGeo(root)


# Buttons
# Create a container frame in row=1, column=0
button_frame = tk.Frame(top_frame)
button_frame.grid(row=1, column=0, padx=10, pady=10)

# Put all three buttons inside this frame, side by side using grid
load_button = tk.Button(button_frame, text="Load Folder", command=load_folder)
load_button.grid(row=0, column=0, padx=5)

coord_button = tk.Button(button_frame, text="Coordination based analysis", command=open_geo_window)
coord_button.grid(row=1, column=0, padx=5)

time_button = tk.Button(button_frame, text="Time based analysis", command=open_range_calendar)
time_button.grid(row=2, column=0, padx=5)

# --- Second Fitting Method Selection Dropdown ---
fitting_frame_4 = ttk.LabelFrame(button_frame, text="Select siking energy channel")
fitting_frame_4.grid(row=3, column=0, padx=10, pady=10, sticky="n")


fitting_frame_5 = ttk.LabelFrame(button_frame, text="Calibration method")
fitting_frame_5.grid(row=4, column=0, padx=10, pady=10, sticky="n")



loc_methods = ["Bi_214 Energy Channel", "Cs_137 Energy Channel", "Be_7 Energy Channel"]
selected_method_4 = tk.StringVar(value=loc_methods[0])

fitting_dropdown_4 = ttk.Combobox(
    fitting_frame_4,
    textvariable=selected_method_4,
    values=loc_methods,
    state="readonly",
    width=30
)
fitting_dropdown_4.grid(padx=10, pady=5)

def confirm_selection_4():
    print(f"Selected method 4: {selected_method_4.get()}")

ttk.Button(fitting_frame_4, text="Confirm Selection", command=confirm_selection_4).grid(pady=5)

loc_methods = ["MARIS Library", "RS Library"]
selected_method_5 = tk.StringVar(value=loc_methods[0])

fitting_dropdown_5 = ttk.Combobox(
    fitting_frame_5,
    textvariable=selected_method_5,
    values=loc_methods,
    state="readonly",
    width=30
)
fitting_dropdown_5.grid(padx=10, pady=5)

def confirm_selection_5():
    print(f"Selected method 5: {selected_method_5.get()}")

ttk.Button(fitting_frame_5, text="Confirm Selection", command=confirm_selection_5).grid(pady=5)













#help_button = tk.Button(top_frame, text="Help", command=show_help)
#help_button.grid(row=2, column=1, padx=10, pady=10)


root.mainloop()


