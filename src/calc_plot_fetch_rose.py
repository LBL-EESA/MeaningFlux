# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:05:45 2022

@author: Leila Hernadez Rodriguez 
Lawrence Berkeley National Laboratory

Description: plot fetch rose based on FFP estimation of the fetch using Kljun approach. 
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from windrose import WindroseAxes
from tkinter import Tk, Listbox, Button, Scrollbar, Frame, Label, StringVar, OptionMenu, messagebox
from tkinter.ttk import Combobox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fetch_rose_window = None

def calc_plot_fetch_rose(df, inputname_site):
    global fetch_rose_window, fig

    # Convert TIMESTAMP_START to datetime
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    # Check if any column starts with FETCH in the dataframe
    def check_fetch_col():
        fetch_cols = [col for col in df.columns if col.startswith('FETCH')]
        if fetch_cols:
            return fetch_cols
        else:
            fetch_label = input("Enter the column label starting with 'FETCH': ")
            return fetch_label

    # Check if WD is labeled as WD or WD_1_1_1 in the dataframe
    def check_wd_col():
        if 'WD' in df:
            return 'WD'
        elif 'WD_1_1_1' in df:
            return 'WD_1_1_1'
        else:
            wind_label = input("Enter the column label for the wind direction: ")
            return wind_label

    fetch_col = check_fetch_col()  # Use FETCH instead of WS
    wd_col = check_wd_col()
    
    def get_and_plot():
        # Get the selected variables to plot
        fetch_selected = fetch_var_selection.get()
        wd_selected = wd_var_selection.get()
        print(f"Selected variables: FETCH - {fetch_selected}, WD - {wd_selected}")

        # Get the selected date range
        start_date = pd.to_datetime(start_date_combobox.get())
        end_date = pd.to_datetime(end_date_combobox.get())

        # Filter the dataframe by the selected date range
        df_filtered = df[(df['TIMESTAMP_START'] >= start_date) & (df['TIMESTAMP_START'] <= end_date)].copy()

        # Clean the data: replace -9999 with NaN
        df_filtered[fetch_selected] = df_filtered[fetch_selected].replace(['', -9999], np.nan)
        df_filtered[wd_selected] = df_filtered[wd_selected].replace(['', -9999], np.nan)

        # Drop rows where either fetch or wind direction is NaN
        df_filtered.dropna(subset=[fetch_selected, wd_selected], inplace=True)

        # Check if the filtered DataFrame is empty
        if df_filtered.empty:
            messagebox.showwarning("Warning", "No valid data available for the selected range and variables")
            return

        # Prepare data for plotting
        fetch = df_filtered[fetch_selected]
        wd = df_filtered[wd_selected]
        dp = pd.DataFrame({"fetch": fetch, "direction": wd})

        # Create fetch rose
        global fig
        fig = plt.figure(figsize=(8, 8), dpi=80)
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(dp.direction, dp.fetch, normed=True, opening=0.8, edgecolor='white')
        ax.set_title(f'Fetch Rose - {inputname_site}')
        ax.set_legend()

        # Clear the previous plot if any
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_plot():
        # Get the selected variables to plot
        fetch_selected = fetch_var_selection.get()
        wd_selected = wd_var_selection.get()
        # Get the selected date range
        start_date = pd.to_datetime(start_date_combobox.get())
        end_date = pd.to_datetime(end_date_combobox.get())

        # Set up the save path
        save_dir = 'Results/Fetch_rose_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Construct the filename, exclude 'None'
        filename = f"{inputname_site.split('.')[0]}_FETCH_{fetch_selected}_WD_{wd_selected}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
        filepath = os.path.join(save_dir, filename)

        # Save the current plot
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def scale():
        global plot_frame, fetch_var_selection, wd_var_selection, fetch_rose_window, start_date_combobox, end_date_combobox

        if fetch_rose_window is not None and fetch_rose_window.winfo_exists():
            fetch_rose_window.lift()
            return

        fetch_rose_window = Tk()
        fetch_rose_window.title('Fetch Rose Plot')
        fetch_rose_window.geometry("1000x700")

        left_frame = Frame(fetch_rose_window)
        left_frame.pack(side='left', fill='y')

        plot_frame = Frame(fetch_rose_window)
        plot_frame.pack(side='right', fill='both', expand=True)

        # Selector for FETCH variable
        Label(left_frame, text="Select fetch variable:").pack()
        fetch_var_selection = StringVar(left_frame)
        fetch_var_selection.set(fetch_col[0])
        fetch_var_menu = OptionMenu(left_frame, fetch_var_selection, *fetch_col)
        fetch_var_menu.pack()

        # Selector for wind direction variable
        Label(left_frame, text="Select wind direction variable:").pack()
        wd_var_selection = StringVar(left_frame)
        wd_var_selection.set(wd_col)
        wd_var_menu = OptionMenu(left_frame, wd_var_selection, wd_col, 'WD', 'WD_1_1_1')
        wd_var_menu.pack()

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
        
        # Description of what the plot is showing
        Label(left_frame, text=f"FETCH_70, FETCH_80, FETCH_90 represent\n"
                      f"the distances at which 70%, 80%, and 90% of the\n"
                      f"footprint's cumulative probability are reached.\n"
                      f"FETCH_MAX indicates the distance where the\n"
                      f"footprint contribution is the highest.\n").pack()


        # Plot the first variable by default
        fetch_rose_window.after(100, get_and_plot)

        fetch_rose_window.protocol("WM_DELETE_WINDOW", on_closing)

        fetch_rose_window.mainloop()

    def on_closing():
        global fetch_rose_window
        fetch_rose_window.destroy()
        fetch_rose_window = None

    scale()
