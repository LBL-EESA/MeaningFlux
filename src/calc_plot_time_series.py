# -*- coding: utf-8 -*-
"""
Author: Leila Hernandez, LBNL
Date: June 14, 2024
Description: Function to plot time series data. Users can select one or two variables to plot from a list and view the corresponding 
             time series plot within the same window. The second variable, if selected, will be plotted on a secondary Y-axis. 
             Users can modify the date range to display in the plot and save the plot to a specified directory.
"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
#from windrose import WindroseAxes
from IPython.display import Image
from pandas import read_csv, DataFrame, Grouper
from matplotlib.dates import DateFormatter
from tkinter import IntVar

# Print Parameters
np.set_printoptions(precision=2)

label_size = 15
ticks_size = 20

time_series_window = None  # Global variable for the time series window
plot_type = 'time_series'  # Default plot type
legend_var = None  # Global variable for legend checkbox

# Define the units for the variables using LaTeX formatting (Units as in AmeriFlux variables)
variable_units = {
    'CO2': r'$\mu mol \, CO_2 \, mol^{-1}$',
    'H2O': r'$mmol \, H_2O \, mol^{-1}$',
    'CH4': r'$nmol \, CH_4 \, mol^{-1}$',
    'NO': r'$nmol \, NO \, mol^{-1}$',
    'NO2': r'$nmol \, NO_2 \, mol^{-1}$',
    'N2O': r'$nmol \, N_2O \, mol^{-1}$',
    'O3': r'$nmol \, O_3 \, mol^{-1}$',
    'FC': r'$\mu mol \, CO_2 \, m^{-2} \, s^{-1}$',
    'FCH4': r'$nmol \, CH_4 \, m^{-2} \, s^{-1}$',
    'FNO': r'$nmol \, NO \, m^{-2} \, s^{-1}$',
    'FNO2': r'$nmol \, NO_2 \, m^{-2} \, s^{-1}$',
    'FN2O': r'$nmol \, N_2O \, m^{-2} \, s^{-1}$',
    'FO3': r'$nmol \, O_3 \, m^{-2} \, s^{-1}$',
    'SC': r'$\mu mol \, CO_2 \, m^{-2} \, s^{-1}$',
    'SCH4': r'$nmol \, CH_4 \, mol^{-1}$',
    'SNO': r'$nmol \, NO \, mol^{-1}$',
    'SNO2': r'$nmol \, NO_2 \, mol^{-1}$',
    'SN2O': r'$nmol \, N_2O \, mol^{-1}$',
    'SO3': r'$nmol \, O_3 \, mol^{-1}$',
    'G': r'$W \, m^{-2}$',
    'H': r'$W \, m^{-2}$',
    'LE': r'$W \, m^{-2}$',
    'SG': r'$W \, m^{-2}$',
    'SH': r'$W \, m^{-2}$',
    'SLE': r'$W \, m^{-2}$',
    'SB': r'$W \, m^{-2}$',
    'WD': r'$degrees$',
    'WS': r'$m \, s^{-1}$',
    'WS_MAX': r'$m \, s^{-1}$',
    'USTAR': r'$m \, s^{-1}$',
    'ZL': r'$adimensional$',
    'TAU': r'$Kg \, m^{-1} \, s^{-2}$',
    'MO_LENGTH': r'$m$',
    'U_SIGMA': r'$m \, s^{-1}$',
    'V_SIGMA': r'$m \, s^{-1}$',
    'W_SIGMA': r'$m \, s^{-1}$',
    'PA': r'$kPa$',
    'RH': r'$%$',
    'TA': r'$^{\circ}C$',
    'VPD': r'$hPa$',
    'T_DP': r'$^{\circ}C$',
    'T_SONIC': r'$^{\circ}C$',
    'T_SONIC_SIGMA': r'$^{\circ}C$',
    'PBLH': r'$m$',
    'SWC': r'$%$',
    'TS': r'$^{\circ}C$',
    'WATER_TABLE_DEPTH': r'$m$',
    'ALB': r'$%$',
    'APAR': r'$\mu mol \, m^{-2} \, s^{-1}$',
    'FAPAR': r'$%$',
    'FIPAR': r'$%$',
    'NETRAD': r'$W \, m^{-2}$',
    'PPFD_IN': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'PPFD_OUT': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'PPFD_BC_IN': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'PPFD_BC_OUT': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'PPFD_DIF': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'PPFD_DIR': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SW_IN': r'$W \, m^{-2}$',
    'SW_OUT': r'$W \, m^{-2}$',
    'SW_BC_IN': r'$W \, m^{-2}$',
    'SW_BC_OUT': r'$W \, m^{-2}$',
    'SW_DIF': r'$W \, m^{-2}$',
    'SW_DIR': r'$W \, m^{-2}$',
    'LW_IN': r'$W \, m^{-2}$',
    'LW_OUT': r'$W \, m^{-2}$',
    'LW_BC_IN': r'$W \, m^{-2}$',
    'LW_BC_OUT': r'$W \, m^{-2}$',
    'SPEC_RED_IN': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SPEC_RED_OUT': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SPEC_RED_REFL': r'$adimensional$',
    'SPEC_NIR_IN': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SPEC_NIR_OUT': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SPEC_NIR_REFL': r'$adimensional$',
    'SPEC_PRI_TGT_IN': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SPEC_PRI_TGT_OUT': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SPEC_PRI_TGT_REFL': r'$adimensional$',
    'SPEC_PRI_REF_IN': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SPEC_PRI_REF_OUT': r'$\mu mol \, Photon \, m^{-2} \, s^{-1}$',
    'SPEC_PRI_REF_REFL': r'$adimensional$',
    'NDVI': r'$adimensional$',
    'PRI': r'$adimensional$',
    'R_UVA': r'$W \, m^{-2}$',
    'R_UVB': r'$W \, m^{-2}$',
    'P': r'$mm$',
    'P_RAIN': r'$mm$',
    'P_SNOW': r'$mm$',
    'D_SNOW': r'$cm$',
    'RUNOFF': r'$mm$',
    'DBH': r'$cm$',
    'LEAF_WET': r'$%$',
    'SAP_DT': r'$^{\circ}C$',
    'SAP_FLOW': r'$mmol \, H_2O \, m^{-2} \, s^{-1}$',
    'STEMFLOW': r'$mm$',
    'THROUGHFALL': r'$mm$',
    'T_BOLE': r'$^{\circ}C$',
    'T_CANOPY': r'$^{\circ}C$',
    'NEE': r'$\mu mol \, CO_2 \, m^{-2} \, s^{-1}$',
    'RECO': r'$\mu mol \, CO_2 \, m^{-2} \, s^{-1}$',
    'GPP': r'$\mu mol \, CO_2 \, m^{-2} \, s^{-1}$'
}


def get_units(variable_name):
    key = variable_name.split('_1')[0]
    return variable_units.get(key, '')

def calc_plot_time_series(df, inputname_site):  
    from tkinter import Tk, Listbox, Button, Scrollbar, Frame, Label, StringVar, OptionMenu, Scale, HORIZONTAL, messagebox, Checkbutton
    from tkinter.ttk import Combobox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    global time_series_window, plot_type, legend_var

    if df is None or df.empty:
        messagebox.showwarning("Warning", "Load the data first")
        return

    if time_series_window is not None and time_series_window.winfo_exists():
        time_series_window.destroy()

    # Convert TIMESTAMP_START to datetime
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    # Columns labels to display in the scrollbar 
    df_head = sorted(df.columns)

    def get_and_plot():
        global fig  
        global legend_var  
    
        primary_var = primary_var_selection.get()
        secondary_var = secondary_var_selection.get()
        start_date = start_date_combobox.get()
        end_date = end_date_combobox.get()

        df_filtered = df[(df['TIMESTAMP_START'] >= start_date) & (df['TIMESTAMP_START'] <= end_date)]

        if plot_type == 'daily_average':
            # Use numeric_only=True to avoid issues with datetime fields
            df_filtered = df_filtered.resample('D', on='TIMESTAMP_START').mean(numeric_only=True).reset_index()

        # Clean the data: replace blanks and -9999 with NaN
        df_filtered[primary_var] = df_filtered[primary_var].replace(['', -9999], np.nan)
        if secondary_var != 'None':
            df_filtered[secondary_var] = df_filtered[secondary_var].replace(['', -9999], np.nan)

        # Plot the graph
        y1 = df_filtered[primary_var]
        x = df_filtered['TIMESTAMP_START']

        fig, ax1 = plt.subplots()

        ax1.plot(x, y1, 'b-', label=primary_var)
        ax1.set_xlabel('Date')

        primary_units = get_units(primary_var)
        primary_label = f"{primary_var} ({primary_units})" if primary_units else f"{primary_var}"
        ax1.set_ylabel(primary_label, color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        myFmt = DateFormatter("%Y/%m/%d")
        ax1.xaxis.set_major_formatter(myFmt)
        fig.autofmt_xdate()

        # Calculate statistics for primary variable excluding NaN values
        primary_mean = np.nanmean(y1)
        primary_max = np.nanmax(y1)
        primary_min = np.nanmin(y1)
        primary_stats_text = f'{primary_var}:\nMean: {primary_mean:.2f}, Max: {primary_max:.2f}, Min: {primary_min:.2f}'

        stats_text = primary_stats_text

        if secondary_var != 'None':
            y2 = df_filtered[secondary_var]
            ax2 = ax1.twinx()

            secondary_units = get_units(secondary_var)
            secondary_label = f"{secondary_var} ({secondary_units})" if secondary_units else f"{secondary_var}"
            ax2.plot(x, y2, 'r-', label=secondary_var)
            ax2.set_ylabel(secondary_label, color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            # Calculate statistics for secondary variable excluding NaN values
            secondary_mean = np.nanmean(y2)
            secondary_max = np.nanmax(y2)
            secondary_min = np.nanmin(y2)
            secondary_stats_text = f'{secondary_var}:\nMean: {secondary_mean:.2f}, Max: {secondary_max:.2f}, Min: {secondary_min:.2f}'

            stats_text += f'\n\n{secondary_stats_text}'

        ax1.set_title(inputname_site)
        ax1.grid(color='0.95')

        # Annotate the plot with mean, max, and min values
        textstr = f'{primary_stats_text}\n{secondary_stats_text if secondary_var != "None" else ""}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax1.text(0.05, -0.25, textstr, transform=ax1.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)

        # Clear the previous plot if any
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        fig.tight_layout()

        fig.subplots_adjust(bottom=0.2)
        
        # Display the statistics
        stats_label.config(text=stats_text)

    def save_plot():
        global fig
        primary_var = primary_var_selection.get()
        secondary_var = secondary_var_selection.get()
        start_date = start_date_combobox.get()
        end_date = end_date_combobox.get()

        save_dir = 'Results/Time_series_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        variables = f"{primary_var}_{secondary_var}" if secondary_var != 'None' else f"{primary_var}"
        plot_type_str = plot_type.replace('_', ' ').capitalize()
        filename = f"{inputname_site.split('.')[0]}_{variables}_{start_date}_{end_date}_{plot_type_str}.png"
        filepath = os.path.join(save_dir, filename)

        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def on_closing():
        global time_series_window
        time_series_window.destroy()
        time_series_window = None

    def scale():
        global plot_frame, primary_var_selection, secondary_var_selection, start_date_combobox, end_date_combobox, fig
        global time_series_window, plot_type

        time_series_window = Tk()
        time_series_window.title('Time series')
        time_series_window.geometry("1000x700")

        time_series_window.protocol("WM_DELETE_WINDOW", on_closing)

        left_frame = Frame(time_series_window)
        left_frame.pack(side='left', fill='y')

        plot_frame = Frame(time_series_window)
        plot_frame.pack(side='right', fill='both', expand=True)

        # Buttons for selecting plot type
        Label(left_frame, text="Select plot type:").pack()
        time_series_button = Button(left_frame, text="Time Series", command=lambda: set_plot_type('time_series'))
        time_series_button.pack()
        daily_average_button = Button(left_frame, text="Daily Average", command=lambda: set_plot_type('daily_average'))
        daily_average_button.pack()

        # Selector for primary variable
        Label(left_frame, text="Select primary variable:").pack()
        primary_var_selection = StringVar(left_frame)
        primary_var_selection.set(df_head[0])
        primary_var_menu = OptionMenu(left_frame, primary_var_selection, *df_head)
        primary_var_menu.pack()

        # Selector for secondary variable
        Label(left_frame, text="Select secondary variable:").pack()
        secondary_var_selection = StringVar(left_frame)
        secondary_var_selection.set('None')
        secondary_var_menu = OptionMenu(left_frame, secondary_var_selection, 'None', *df_head)
        secondary_var_menu.pack()

        # Date range selection
        Label(left_frame, text="Select start date:").pack()
        start_date_combobox = Combobox(left_frame, values=list(df['TIMESTAMP_START'].astype(str)))
        start_date_combobox.set(str(df['TIMESTAMP_START'].min()))
        start_date_combobox.pack()

        Label(left_frame, text="Select end date:").pack()
        end_date_combobox = Combobox(left_frame, values=list(df['TIMESTAMP_START'].astype(str)))
        end_date_combobox.set(str(df['TIMESTAMP_START'].max()))
        end_date_combobox.pack()

        # Button for plotting
        selectbutton = Button(left_frame, text="Plot", command=get_and_plot)
        selectbutton.pack()

        # Button for saving the plot
        savebutton = Button(left_frame, text="Save Plot", command=save_plot)
        savebutton.pack()
        
        global stats_label
        
        # Label to display statistics
        stats_label = Label(left_frame, text="", justify='left')
        stats_label.pack()

        # Automatically plot the first variable on startup
        time_series_window.after(1000, get_and_plot)
        time_series_window.mainloop()

    def set_plot_type(ptype):
        global plot_type
        plot_type = ptype
        get_and_plot()

    # Call the scale function to open the window and initialize the plot
    scale()

def escape_latex(s):
    return s.replace('_', '\_')

