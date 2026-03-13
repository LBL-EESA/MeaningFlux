#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Leila Hernandez, LBNL
Date: June 17, 2024
Description: This script provides a function to plot density and scatter plots of selected variables 
             from a given dataset using a graphical user interface (GUI) with tkinter. Users can select 
             two variables to plot, choose the time frame, and save the generated plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, StringVar, OptionMenu, Button, Frame, Label, messagebox
from tkinter.ttk import Combobox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import pearsonr, spearmanr

density_scatter_window = None  # Initialize density_scatter_window as None globally

def calc_plot_density_and_scatter(df, inputname_site):  # Function to open a new window
    global density_scatter_window, fig

    # Check if the dataframe is provided
    if df is None or df.empty:
        messagebox.showwarning("Warning", "Load the data first")
        return

    # Ensure window can be reopened after being closed
    if density_scatter_window is not None and density_scatter_window.winfo_exists():
        density_scatter_window.lift()
        return

    # Convert TIMESTAMP_START to datetime
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    # Columns labels to display in the options
    df_head = sorted(df.columns)

    def get_and_plot():
        global fig  # Make fig accessible in save_plot

        # Get the selected variables to plot
        var1 = var1_selection.get()
        var2 = var2_selection.get()
        print(f"Variable 1: {var1}")
        print(f"Variable 2: {var2}")

        # Get the selected date range
        start_date = start_date_combobox.get()
        end_date = end_date_combobox.get()
        print(f"Date range: {start_date} to {end_date}")

        # Filter the dataframe by the selected date range
        df_filtered = df[(df['TIMESTAMP_START'] >= start_date) & (df['TIMESTAMP_START'] <= end_date)].copy()

        # Clean the data: replace blanks and -9999 with NaN
        df_filtered[var1].replace(['', -9999], np.nan, inplace=True)
        df_filtered[var2].replace(['', -9999], np.nan, inplace=True)

        # Drop rows with NaN values in the selected columns
        df_filtered.dropna(subset=[var1, var2], inplace=True)

        # Calculate Pearson correlation
        pearson_corr, pearson_p = pearsonr(df_filtered[var1], df_filtered[var2])
        pearson_text = f'Pearson correlation: {pearson_corr:.2f} (p-value: {pearson_p:.2e})'

        # Calculate Spearman correlation
        spearman_corr, spearman_p = spearmanr(df_filtered[var1], df_filtered[var2])
        spearman_text = f'Spearman correlation: {spearman_corr:.2f} (p-value: {spearman_p:.2e})'

        # Plotting the density and scatter plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Density plot for Variable 1
        df_filtered[var1].plot(kind='density', ax=ax1, color='blue', label=var1)
        df_filtered[var2].plot(kind='density', ax=ax1, color='red', label=var2)
        ax1.set_title(f'Density Plot of {var1} and {var2}')
        ax1.grid(True)
        ax1.legend()

        # Scatter plot between Variable 1 and Variable 2
        ax2.scatter(df_filtered[var1], df_filtered[var2], color='green')
        ax2.set_title(f'Scatter Plot of {var1} vs {var2}')
        ax2.set_xlabel(var1)
        ax2.set_ylabel(var2)
        ax2.grid(True)
        
        # Annotate the scatter plot with correlation coefficients
        textstr = f'{pearson_text}\n{spearman_text}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax2.text(0.05, -0.15, textstr, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='bottom', bbox=props)

        # Add overall title to the figure
        fig.suptitle(f'Density and Scatter Plots for {inputname_site}', fontsize=16)

        # Clear the previous plot if any
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Display correlation metrics in a Label widget
        correlation_text = f'{pearson_text}\n{spearman_text}'
        correlation_label.config(text=correlation_text)

    def save_plot():
        global fig
        # Get the selected variables to plot
        var1 = var1_selection.get()
        var2 = var2_selection.get()
        start_date = start_date_combobox.get()
        end_date = end_date_combobox.get()

        # Set up the save path
        save_dir = 'Results/Density_scatter_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Construct the filename
        filename = f"{inputname_site}_{var1}_{var2}_{start_date}_{end_date}_density_scatter.png"
        filepath = os.path.join(save_dir, filename)

        # Save the current plot
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def on_closing():
        global density_scatter_window
        density_scatter_window.destroy()
        density_scatter_window = None

    def scale():
        global plot_frame, var1_selection, var2_selection, start_date_combobox, end_date_combobox, fig
        global density_scatter_window

        density_scatter_window = Tk()
        density_scatter_window.title(f'Density and Scatter Plots')
        density_scatter_window.geometry("1000x700")

        # Add this line to reset the variable when the window is closed
        density_scatter_window.protocol("WM_DELETE_WINDOW", on_closing)

        left_frame = Frame(density_scatter_window)
        left_frame.pack(side='left', fill='y')

        plot_frame = Frame(density_scatter_window)
        plot_frame.pack(side='right', fill='both', expand=True)

        # Selector for variable 1
        Label(left_frame, text="Select Variable 1:").pack()
        var1_selection = StringVar(left_frame)
        var1_selection.set(df_head[0])
        var1_menu = OptionMenu(left_frame, var1_selection, *df_head)
        var1_menu.pack()

        # Selector for variable 2
        Label(left_frame, text="Select Variable 2:").pack()
        var2_selection = StringVar(left_frame)
        var2_selection.set(df_head[1])
        var2_menu = OptionMenu(left_frame, var2_selection, *df_head)
        var2_menu.pack()

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
        plot_button = Button(left_frame, text="Plot", command=get_and_plot)
        plot_button.pack()

        # Button for saving the plot
        save_button = Button(left_frame, text="Save Plot", command=save_plot)
        save_button.pack()

        # Add this line to initialize the correlation label
        global correlation_label
        correlation_label = Label(left_frame, text="", justify='left')
        correlation_label.pack()

        density_scatter_window.after(1000, get_and_plot)  # Automatically plot the first variables on startup
        density_scatter_window.mainloop()

    scale()

