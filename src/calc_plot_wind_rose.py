# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:05:45 2022

@author: Leila Hernadez Rodriguez 
Lawrence Berkeley National Laboratory

Description: plot wind rose based on user-selected wind speed and wind direction variables 
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

wind_rose_window = None

def calc_plot_wind_rose(df, inputname_site):
    global wind_rose_window, fig

    # Convert TIMESTAMP_START to datetime
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    # Check if WS is labeled as WS, WS_F, or WS_1_1_1 in the dataframe
    def check_ws_col():
        if 'WS' in df:
            return 'WS'
        elif 'WS_F' in df:
            return 'WS_F'
        elif 'WS_1_1_1' in df:
            return 'WS_1_1_1'
        else:
            wind_label = input("Enter the column label for the wind speed: ")
            return wind_label

    # Check if WD is labeled as WD or WD_1_1_1 in the dataframe
    def check_wd_col():
        if 'WD' in df:
            return 'WD'
        elif 'WD_1_1_1' in df:
            return 'WD_1_1_1'
        else:
            wind_label = input("Enter the column label for the wind direction: ")
            return wind_label

    ws_col = check_ws_col()
    wd_col = check_wd_col()
    
    def get_and_plot():
        # Get the selected variables to plot
        ws_selected = ws_var_selection.get()
        wd_selected = wd_var_selection.get()
        print(f"Selected variables: WS - {ws_selected}, WD - {wd_selected}")

        # Get the selected date range
        start_date = pd.to_datetime(start_date_combobox.get())
        end_date = pd.to_datetime(end_date_combobox.get())

        # Filter the dataframe by the selected date range
        df_filtered = df[(df['TIMESTAMP_START'] >= start_date) & (df['TIMESTAMP_START'] <= end_date)].copy()

        # Clean the data: replace -9999 with NaN
        df_filtered[ws_selected] = df_filtered[ws_selected].replace(['', -9999], np.nan)
        df_filtered[wd_selected] = df_filtered[wd_selected].replace(['', -9999], np.nan)

        # Drop rows where either wind speed or wind direction is NaN
        df_filtered.dropna(subset=[ws_selected, wd_selected], inplace=True)

        # Check if the filtered DataFrame is empty
        if df_filtered.empty:
            messagebox.showwarning("Warning", "No valid data available for the selected range and variables")
            return

        # Prepare data for plotting
        ws = df_filtered[ws_selected]
        wd = df_filtered[wd_selected]
        dp = pd.DataFrame({"speed": ws, "direction": wd})

        # Create wind rose
        global fig
        fig = plt.figure(figsize=(8, 8), dpi=80)
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(dp.direction, dp.speed, normed=True, opening=0.8, edgecolor='white')
        ax.set_title(f'Wind Rose - {inputname_site}')
        ax.set_legend()

        # Clear the previous plot if any
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_plot():
        # Get the selected variables to plot
        ws_selected = ws_var_selection.get()
        wd_selected = wd_var_selection.get()
        # Get the selected date range
        start_date = pd.to_datetime(start_date_combobox.get())
        end_date = pd.to_datetime(end_date_combobox.get())

        # Set up the save path
        save_dir = 'Results/Wind_rose_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Construct the filename, exclude 'None'
        filename = f"{inputname_site.split('.')[0]}_WS_{ws_selected}_WD_{wd_selected}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png"
        filepath = os.path.join(save_dir, filename)

        # Save the current plot
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def scale():
        global plot_frame, ws_var_selection, wd_var_selection, wind_rose_window, start_date_combobox, end_date_combobox

        if wind_rose_window is not None and wind_rose_window.winfo_exists():
            wind_rose_window.lift()
            return

        wind_rose_window = Tk()
        wind_rose_window.title('Wind Rose Plot')
        wind_rose_window.geometry("1000x700")

        left_frame = Frame(wind_rose_window)
        left_frame.pack(side='left', fill='y')

        plot_frame = Frame(wind_rose_window)
        plot_frame.pack(side='right', fill='both', expand=True)

        # Selector for wind speed variable
        Label(left_frame, text="Select wind speed variable:").pack()
        ws_var_selection = StringVar(left_frame)
        ws_var_selection.set(ws_col)
        ws_var_menu = OptionMenu(left_frame, ws_var_selection, ws_col, 'WS', 'WS_F', 'WS_1_1_1')
        ws_var_menu.pack()

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
        Label(left_frame, text=f"The wind rose visualizes the distribution of wind \n"
                               f"direction and speed, with bars representing the \n"
                               f"frequency of wind speed in different directions.").pack()

        # Plot the first variable by default
        wind_rose_window.after(100, get_and_plot)

        wind_rose_window.protocol("WM_DELETE_WINDOW", on_closing)

        wind_rose_window.mainloop()

    def on_closing():
        global wind_rose_window
        wind_rose_window.destroy()
        wind_rose_window = None
        
        

    scale()
