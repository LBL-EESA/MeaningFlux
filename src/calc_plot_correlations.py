# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:59:45 2022

Estimates correlations among all Ameriflux variables 

@author: Leila Hernadez Rodriguez 
Lawrence Berkeley National Laboratory
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tkinter import Tk, Listbox, Button, Scrollbar, Frame, Label, StringVar, OptionMenu, messagebox, Checkbutton, IntVar
from tkinter.ttk import Combobox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

correlation_window = None  # Initialize correlation_window as None globally

def calc_plot_correlations(df, inputname_site):
    global correlation_window, fig, show_values_var, correlation_method_var

    # Convert TIMESTAMP_START to datetime
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    def get_and_plot():
        global fig

        # Debugging output to check the value of show_values_var
        print(f"Show values: {show_values_var.get()}")

        # Close the previous figure if it exists
        try:
            plt.close(fig)
        except NameError:
            pass

        # Get the selected columns to plot
        selected_columns = [columns_listbox.get(i) for i in columns_listbox.curselection()]
        if not selected_columns:
            selected_columns = df.columns.tolist()[:10]  # Default to first 10 columns if none selected
        print(f"Selected columns: {selected_columns}")

        # Get the selected date range
        start_date = pd.to_datetime(start_date_combobox.get())
        end_date = pd.to_datetime(end_date_combobox.get())

        # Filter the dataframe by the selected date range
        df_filtered = df[(df['TIMESTAMP_START'] >= start_date) & (df['TIMESTAMP_START'] <= end_date)].copy()

        # Prepare data for plotting
        data = df_filtered[selected_columns].copy()

        # Get the selected correlation method
        correlation_method = correlation_method_var.get()
        print(f"Correlation method: {correlation_method}")

        # Create correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        annot = bool(show_values_var.get())  # Convert to boolean
        sns.heatmap(data.corr(method=correlation_method), annot=annot, ax=ax, cmap="coolwarm")
        ax.set_title(f'Correlation Heatmap ({correlation_method.capitalize()}) - {inputname_site}')
        
        # Clear the previous plot if any
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_plot():
        global fig

        # Get the selected columns to plot
        selected_columns = [columns_listbox.get(i) for i in columns_listbox.curselection()]
        if not selected_columns:
            selected_columns = df.columns.tolist()[:10]  # Default to first 10 columns if none selected
        # Get the selected date range
        start_date = pd.to_datetime(start_date_combobox.get())
        end_date = pd.to_datetime(end_date_combobox.get())

        # Get the selected correlation method
        correlation_method = correlation_method_var.get()

        # Set up the save path
        save_dir = 'Results/Correlation_heatmap_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Construct the filename
        filename = f"{inputname_site.split('.')[0]}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_correlation_heatmap_{correlation_method}.png"
        filepath = os.path.join(save_dir, filename)

        # Save the current plot
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def on_closing():
        global correlation_window
        correlation_window.destroy()
        correlation_window = None

    def scale():
        global plot_frame, start_date_combobox, end_date_combobox, columns_listbox, show_values_var, correlation_window, correlation_method_var

        if correlation_window is not None and correlation_window.winfo_exists():
            correlation_window.lift()
            return

        correlation_window = Tk()
        correlation_window.title('Correlation Heatmap Plot')
        correlation_window.geometry("1200x800")

        # Add this line to reset the variable when the window is closed
        correlation_window.protocol("WM_DELETE_WINDOW", on_closing)

        left_frame = Frame(correlation_window)
        left_frame.pack(side='left', fill='y')

        plot_frame = Frame(correlation_window)
        plot_frame.pack(side='right', fill='both', expand=True)

        # Selector for columns
        Label(left_frame, text="Select columns (uncheck to exclude):").pack()
        columns_listbox = Listbox(left_frame, selectmode='multiple', exportselection=0)
        for col in df.columns:
            if col not in ['TIMESTAMP_START', 'TIMESTAMP_END', 'DATESTAMP_START']:
                columns_listbox.insert('end', col)
                if columns_listbox.size() <= 10:
                    columns_listbox.selection_set(columns_listbox.size() - 1)  # Select first 10 columns by default
        columns_listbox.pack(fill='both', expand=True)

        # Date range selection
        Label(left_frame, text="Select start date:").pack()
        start_date_combobox = Combobox(left_frame, values=list(df['TIMESTAMP_START'].astype(str)))
        start_date_combobox.set(str(df['TIMESTAMP_START'].min()))
        start_date_combobox.pack()

        Label(left_frame, text="Select end date:").pack()
        end_date_combobox = Combobox(left_frame, values=list(df['TIMESTAMP_START'].astype(str)))
        end_date_combobox.set(str(df['TIMESTAMP_START'].max()))
        end_date_combobox.pack()

        # Dropdown to select correlation method
        Label(left_frame, text="Select correlation method:").pack()
        correlation_method_var = StringVar(value='pearson')  # Default to Pearson
        correlation_method_dropdown = OptionMenu(left_frame, correlation_method_var, 'Pearson (linear)', 'Spearman (monotonic)')
        correlation_method_dropdown.pack()
        correlation_method_dropdown.config(width=20)  # Ensure visibility of the default option

        # Checkbox to show or hide values in cells
        show_values_var = IntVar(value=1)  # Default to showing values
        show_values_checkbutton = Checkbutton(left_frame, text="Show values", variable=show_values_var, command=get_and_plot)
        show_values_checkbutton.pack()

        # Button for plotting
        selectbutton = Button(left_frame, text="Plot", command=get_and_plot)
        selectbutton.pack()

        # Button for saving the plot
        savebutton = Button(left_frame, text="Save Plot", command=save_plot)
        savebutton.pack()

        # Plot the heatmap by default
        correlation_window.after(100, get_and_plot)

        correlation_window.mainloop()

    scale()
