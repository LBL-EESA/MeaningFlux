# -*- coding: utf-8 -*-
"""
Author: Leila Hernandez, LBNL
Date: June 14, 2024
Description: 
    
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Listbox, Button, Scrollbar, Frame, Label, StringVar, OptionMenu, messagebox
from tkinter.ttk import Combobox


# Print Parameters
np.set_printoptions(precision=2)

label_size = 15
ticks_size = 20

# Define the units for the variables using LaTeX formatting
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

daily_avg_window = None

def get_units(variable_name):
    key = variable_name.split('_1')[0]
    return variable_units.get(key, '')

def calc_plot_daily_avg(df, inputname_site):  # Function to open a new window
    from tkinter import Tk, Listbox, Button, Scrollbar, Frame, Label, StringVar, OptionMenu, messagebox
    from tkinter.ttk import Combobox
    from matplotlib.figure import Figure
    from tkinter import TclError
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import calendar

    global daily_avg_window, fig

    # Check if the dataframe is provided
    if df is None or df.empty:
        messagebox.showwarning("Warning", "Load the data first")
        return

    # Ensure window can be reopened after being closed
    try:
        if daily_avg_window is not None and daily_avg_window.winfo_exists():
            daily_avg_window.lift()
            return
    except TclError:
        pass

    daily_avg_window = Tk()
    daily_avg_window.title('Daily Average Plot')
    daily_avg_window.geometry("1000x700")

    # Convert TIMESTAMP_START to datetime
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    # Columns labels to display in the scrollbar 
    df_head = sorted(df.columns)

    def get_and_plot():
        # Get the selected variable to plot
        userline = primary_var_selection.get()
        print(f"Selected variable: {userline}")

        # Get the selected date range
        start_date = pd.to_datetime(start_date_combobox.get())
        end_date = pd.to_datetime(end_date_combobox.get())

        # Filter the dataframe by the selected date range
        df_filtered = df[(df['TIMESTAMP_START'] >= start_date) & (df['TIMESTAMP_START'] <= end_date)]

        # Clean the data: replace -9999 with NaN
        df_filtered.loc[:, userline] = df_filtered[userline].replace(['', -9999], np.nan)

        # Prepare data for plotting
        df_filtered['Hour'] = df_filtered['TIMESTAMP_START'].dt.hour
        df_filtered['Month'] = df_filtered['TIMESTAMP_START'].dt.month

        # Create subplots for each month
        global fig
        fig, axs = plt.subplots(3, 4, figsize=(20, 15), sharey=True, sharex=True)
        fig.suptitle(f'{inputname_site} - Daily Average, Min, and Max Values for {userline}', fontsize=16)

        for month in range(1, 13):
            ax = axs[(month-1)//4, (month-1)%4]
            monthly_data = df_filtered[df_filtered['Month'] == month]

            if not monthly_data.empty:
                # Group data by hour and calculate mean, min, max, and standard deviation
                daily_stats = monthly_data.groupby('Hour')[userline].agg(['mean', 'min', 'max', 'std'])

                # Plot mean with variability bands
                ax.plot(daily_stats.index, daily_stats['mean'], 'k-', label='Mean', linewidth=2)
                ax.fill_between(daily_stats.index, daily_stats['mean'] - daily_stats['std'], daily_stats['mean'] + daily_stats['std'],
                                color='gray', alpha=0.3, label='Std Dev')
                ax.plot(daily_stats.index, daily_stats['min'], 'b--', label='Min', linewidth=2)
                ax.plot(daily_stats.index, daily_stats['max'], 'r--', label='Max', linewidth=2)
                ax.set_title(calendar.month_name[month])
                ax.set_xticks([0, 6, 12, 18, 23])
                ax.set_xticklabels(['0', '6', '12', '18', '23'])
                ax.grid(True)

        # Set a single x-axis label for all subplots
        fig.text(0.5, 0.04, 'Hour', ha='center', fontsize=12)

        # Set y-axis labels for each column of subplots
        for ax in axs[:, 0]:
            ax.set_ylabel(f'{userline} ({get_units(userline)})')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')

        # Clear the previous plot if any
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_plot():
        # Get the selected variable to plot
        userline = primary_var_selection.get()
        # Get the selected date range
        start_date = pd.to_datetime(start_date_combobox.get())
        end_date = pd.to_datetime(end_date_combobox.get())

        # Set up the save path
        save_dir = 'Results/Daily_avg_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Construct the filename, exclude 'None'
        filename = f"{inputname_site.split('.')[0]}_{userline}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
        filepath = os.path.join(save_dir, filename)

        # Save the current plot
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def scale():
        global plot_frame, primary_var_selection, daily_avg_window, start_date_combobox, end_date_combobox, stats_label

        left_frame = Frame(daily_avg_window)
        left_frame.pack(side='left', fill='y')

        plot_frame = Frame(daily_avg_window)
        plot_frame.pack(side='right', fill='both', expand=True)

        # Selector for primary variable
        Label(left_frame, text="Select primary variable:").pack()
        primary_var_selection = StringVar(left_frame)
        primary_var_selection.set(df_head[0])
        primary_var_menu = OptionMenu(left_frame, primary_var_selection, *df_head)
        primary_var_menu.pack()

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

        # Label to display statistics
        stats_label = Label(left_frame, text="", justify='left')
        stats_label.pack()

        # Plot the first variable by default
        daily_avg_window.after(100, get_and_plot)

        daily_avg_window.mainloop()

    scale()
