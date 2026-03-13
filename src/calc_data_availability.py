#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:51:00 2024

Author: Leila C Hernandez Rodriguez 
Lawrence Berkeley National Laboratory, Berkeley, CA, USA (lchernandezrodriguez@lbl.gov)
ORCID: 0000-0001-8830-345X

Description:
Estimates the availability of eddy covariance data, generating an Excel file and a plot for the selected variable.

For data density estimation:
AmeriFlux time is reported in local standard time without Daylight Saving Time. Uneven data density can bias gap-filling, typically with less data at night. Adjusting the U* threshold in EddyPro can reduce gaps by 40%.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Message, Frame, Label, StringVar, OptionMenu, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

time_series_window = None  # Initialize time_series_window as None globally

def calc_data_availability(df, inputname_site):
    global time_series_window, primary_var_selection

    if df is None or df.empty:
        messagebox.showwarning("Warning", "Load the data first")
        return

    df_head = sorted(df.columns)

    output_folder = os.path.join(os.getcwd(), "Results", "data_availability_plots_and_report")
    os.makedirs(output_folder, exist_ok=True)

    output_subfolder = os.path.join(output_folder)
    os.makedirs(output_subfolder, exist_ok=True)

    def calculate_inferred_frequency_and_expected_points(df):
        time_diffs = df['TIMESTAMP_START'].diff().dropna()
        if time_diffs.empty:
            return None, None

        median_time_diff_seconds = time_diffs.median().total_seconds()
        inferred_data_frequency_minutes = median_time_diff_seconds / 60

        return inferred_data_frequency_minutes, time_diffs

    def plot_datapoints_per_timestep(search_variable):
        df['Hour_Minute'] = df['TIMESTAMP_START'].dt.strftime('%H:%M')
        df_filtered = df[(df[search_variable] != -9999) & (~df[search_variable].astype(str).str.strip().eq('')) & (~df[search_variable].isna())]
        datapoints_per_timestep = df_filtered.groupby('Hour_Minute').size()
        mean_value = round(datapoints_per_timestep.mean())
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axhline(y=mean_value, color='black', linestyle='--', label=f'Average:{mean_value}')
        colors = ['green' if val >= mean_value else 'red' for val in datapoints_per_timestep]
        datapoints_per_timestep.plot(kind='bar', color=colors, ax=ax)
        ax.set_title(f'Data density for each timestep for {search_variable} in {inputname_site}')
        ax.set_xlabel('Hour:Minute')
        ax.set_ylabel('Number of Datapoints')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        return fig

    def get_and_plot(search_variable):
        user_variable = [search_variable]
        df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format='%Y%m%d%H%M', errors='coerce')
        df['Year'] = df["TIMESTAMP_START"].dt.year
        df['Month'] = df["TIMESTAMP_START"].dt.month_name()
        monthly_completeness_plot = df.groupby(['Year', 'Month'])[user_variable[0]].apply(lambda x: (~x.astype(str).str.strip().eq('') & (x != -9999) & ~x.isna()).mean() * 100)
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        monthly_completeness_plot = monthly_completeness_plot.reindex(months, level=1)
        color_list = ['red','cyan', 'magenta', 'yellow', 'black', 'lime', 'brown', 'olive', 'pink', 'teal', 'navy', 'maroon', 'gold', 'indigo', 'silver', 'orchid', 'crimson']
        colors = {}
        unique_years = df['Year'].unique()
        for year in unique_years:
            if not pd.isnull(year):
                if year in [2020, 2021, 2022, 2023]:
                    if year == 2020:
                        colors[year] = 'purple'
                    elif year == 2021:
                        colors[year] = 'blue'
                    elif year == 2022:
                        colors[year] = 'darkorange'
                    elif year == 2023:
                        colors[year] = 'green'
                else:
                    if year not in colors:
                        color_index = (len(colors) - 4) % len(color_list)  # Adjust index to avoid reuse of fixed colors
                        colors[year] = color_list[color_index]
                        color_list.pop(color_index)  # Remove the used color from the list

        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_completeness_plot.unstack(level=0).plot(kind='bar', ax=ax, color=[colors.get(year, 'grey') for year in monthly_completeness_plot.unstack(level=0).columns])
        ax.set_ylabel('% of availability')
        ax.set_title(f'Data availability for {search_variable} in {inputname_site}')
        ax.legend(title='Year')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        return fig

    def generate_excel(search_variable):
        user_variable = [search_variable]
        df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format='%Y%m%d%H%M', errors='coerce')
        df['Year'] = df["TIMESTAMP_START"].dt.year
        completeness_df = pd.DataFrame(columns=["File", "Variable", "Year", "Inferred Data Frequency (minutes)", "Start Timestamp", "End Timestamp", "Expected Data Points", "Actual Data Points", "Percentage of Completeness", "Non-consecutive Gaps"])
        for year in df['Year'].unique():
            df_year = df[df['Year'] == year]
            inferred_data_frequency, time_diffs = calculate_inferred_frequency_and_expected_points(df_year)
            if inferred_data_frequency is None:
                completeness_df = pd.concat([completeness_df, pd.DataFrame({
                    "File": os.path.basename(inputname_site),
                    "Variable": user_variable[0],
                    "Year": year,
                    "Inferred Data Frequency (minutes)": "N/A",
                    "Start Timestamp": "N/A",
                    "End Timestamp": "N/A",
                    "Expected Data Points": "N/A",
                    "Actual Data Points": "N/A",
                    "Percentage of Availability": "N/A",
                    "Non-consecutive Gaps": "N/A"
                }, index=[0])], ignore_index=True)
                continue
            expected_data_points = int((df_year['TIMESTAMP_START'].max() - df_year['TIMESTAMP_START'].min()).total_seconds() / (inferred_data_frequency * 60))
            actual_data_points = len(df_year[~df_year[user_variable[0]].astype(str).str.strip().eq('') & (df_year[user_variable[0]] != -9999) & ~df_year[user_variable[0]].isna()])
            completeness_percentage = int(round((actual_data_points / expected_data_points) * 100))
            gaps = time_diffs[time_diffs > pd.Timedelta(minutes=2*inferred_data_frequency)]
            gap_dates = [(df_year.loc[idx, 'TIMESTAMP_START'].strftime('%Y-%m-%d %H:%M:%S'), (df_year.loc[idx, 'TIMESTAMP_START'] + time_diffs.loc[idx]).strftime('%Y-%m-%d %H:%M:%S')) for idx in gaps.index]
            gaps_str = '\n'.join([f"• {start} to {end}" for start, end in gap_dates])
            completeness_df = pd.concat([completeness_df, pd.DataFrame({
                "File": os.path.basename(inputname_site),
                "Variable": user_variable[0],
                "Year": year,
                "Inferred Data Frequency (minutes)": inferred_data_frequency,
                "Start Timestamp": df_year['TIMESTAMP_START'].min(),
                "End Timestamp": df_year['TIMESTAMP_START'].max(),
                "Expected Data Points": expected_data_points,
                "Actual Data Points": actual_data_points,
                "Percentage of Availability": completeness_percentage,
                "Non-consecutive Gaps": gaps_str if gap_dates else "None"
            }, index=[0])], ignore_index=True)
        output_file = os.path.join(output_subfolder, f"{inputname_site}_{search_variable}_data_availability.xlsx")
        completeness_df.to_excel(output_file, index=False)
        print(f"Excel file generated successfully and saved to {output_file}")

    def visualize_data_availability(inputname_site):
        df["TIMESTAMP_START"] = pd.to_datetime(df["TIMESTAMP_START"], format="%Y%m%d%H%M", errors="coerce")
        df["Year"] = df["TIMESTAMP_START"].dt.year
        df["Month"] = df["TIMESTAMP_START"].dt.to_period("M")
    
        data_availability = (
            df.groupby(["Month"])[df_head]
              .apply(lambda x: x.isna().mean() * 100)
              .T
        )
    
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # 0% missing = blue, 100% missing = red
        cmap = plt.get_cmap("RdYlBu_r")
        norm = Normalize(vmin=0, vmax=100)
        cax = ax.imshow(data_availability, cmap=cmap, norm=norm, aspect="auto")
    
        # --- Colorbar placed outside with comfortable padding
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="2.5%", pad=0.40)  # increase pad to move farther right
        cb = fig.colorbar(cax, cax=cbar_ax)
        cb.set_label("Missing Data Percentage")
        cb.set_ticks([0, 25, 50, 75, 100])
    
        # Ticks/labels/title
        ax.set_yticks(range(len(data_availability.index)))
        ax.set_yticklabels(data_availability.index)
        ax.set_xticks(range(len(data_availability.columns)))
        ax.set_xticklabels([str(p) for p in data_availability.columns], rotation=45)
        ax.set_title(f"Missing Data Percentage Over Time\n{inputname_site}")
    
        # Caption under plot
        ax.text(0.5, -0.16, "Dark red = all data missing; darker blue = complete data.",
                transform=ax.transAxes, ha="center", va="top", fontsize=10)
    
        fig.subplots_adjust(bottom=0.22)  # room for caption
        return fig


    def save_plot(plot_type):
        global fig
        if plot_type == 'data_availability':
            plot_file = os.path.join(output_subfolder, f'{inputname_site}_data_availability.png')
        elif plot_type == 'data_completeness':
            plot_file = os.path.join(output_subfolder, f'{inputname_site}_{primary_var_selection.get()}_data_completeness.png')
        elif plot_type == 'data_density':
            plot_file = os.path.join(output_subfolder, f'{inputname_site}_{primary_var_selection.get()}_data_density.png')
        fig.savefig(plot_file)
        print(f"Plot saved to {plot_file}")

    def update_plot(plot_func, *args):
        global fig, canvas
        for widget in plot_frame.winfo_children():
            widget.destroy()
        fig = plot_func(*args)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def check_data_gaps():
        inferred_frequency, time_diffs = calculate_inferred_frequency_and_expected_points(df)
        if inferred_frequency is None:
            gap_message.config(text="Inferred data frequency: N/A\nNo data Available to calculate gaps.")
            return
        gaps = time_diffs[time_diffs > pd.Timedelta(minutes=2*inferred_frequency)]
        gap_info = "No non-consecutive gaps detected." if gaps.empty else f"Non-consecutive gaps detected: {len(gaps)}"
        freq_info = f"Inferred data frequency: {inferred_frequency:.2f} minutes"
        gap_dates = [(df.loc[idx, 'TIMESTAMP_START'].strftime('%Y-%m-%d %H:%M:%S'), (df.loc[idx, 'TIMESTAMP_START'] + time_diffs.loc[idx]).strftime('%Y-%m-%d %H:%M:%S')) for idx in gaps.index]
        gap_message.config(text=f"{freq_info}\n{gap_info}\nGaps:\n" + "\n".join([f"• {start} to {end}" for start, end in gap_dates]))

    def on_closing():
        global time_series_window
        time_series_window.destroy()
        time_series_window = None

    def scale():
        global fig, canvas, plot_frame, gap_message, primary_var_selection, time_series_window
    
        if time_series_window is not None and time_series_window.winfo_exists():
            time_series_window.lift()
            return

        time_series_window = Tk()
        time_series_window.title('Data Availability')
        time_series_window.geometry("1200x900")
        
        # Add this line to reset the variable when the window is closed
        time_series_window.protocol("WM_DELETE_WINDOW", on_closing)
        
        left_frame = Frame(time_series_window)
        left_frame.pack(side='left', fill='y')
        
        plot_frame = Frame(time_series_window)
        plot_frame.pack(side='right', fill='both', expand=True)
                
        # Label for Available data visualization
        Label(left_frame, text="Visualize Available Data for All Variables (% of NaNs).").pack(pady=5)
        
        # Create a frame to hold the visualize and save buttons for Available data
        data_availability_frame = Frame(left_frame)
        data_availability_frame.pack(pady=5)
        Button(data_availability_frame, text="Visualize Available Data", command=lambda: update_plot(visualize_data_availability, inputname_site)).grid(row=0, column=0, padx=(0, 5))
        Button(data_availability_frame, text="Save Plot", command=lambda: save_plot('data_availability')).grid(row=0, column=1)

        # Add a spacer
        Label(left_frame, text="").pack(pady=5)
        
        Label(left_frame, text="Available data in a single variable:").pack()
        primary_var_selection = StringVar(left_frame)
        primary_var_selection.set(df_head[0])
        primary_var_menu = OptionMenu(left_frame, primary_var_selection, *df_head)
        primary_var_menu.pack()
    
        # Create a frame to hold each pair of buttons
        availability_frame = Frame(left_frame)
        availability_frame.pack(pady=5)
        Button(availability_frame, text="Visualize Data Availability", command=lambda: update_plot(get_and_plot, primary_var_selection.get())).grid(row=0, column=0, padx=(0, 5))
        Button(availability_frame, text="Save Plot", command=lambda: save_plot('data_avalability')).grid(row=0, column=1)
        
        density_frame = Frame(left_frame)
        density_frame.pack(pady=5)
        Button(density_frame, text="Visualize Data Density", command=lambda: update_plot(plot_datapoints_per_timestep, primary_var_selection.get())).grid(row=0, column=0, padx=(0, 5))
        Button(density_frame, text="Save Plot", command=lambda: save_plot('data_density')).grid(row=0, column=1)

        Button(left_frame, text="Check Data Gaps", command=check_data_gaps).pack()
        gap_explanation = Message(left_frame, text="Non-consecutive gaps are periods with missing data between timestamps, indicating collection issues.")
        gap_explanation.pack(pady=5)
        
        gap_message = Label(left_frame, text="", wraplength=250, justify="left")
        gap_message.pack(pady=5)
        
        Button(left_frame, text="Generate Report", command=lambda: generate_excel(primary_var_selection.get())).pack()
    
        time_series_window.after(1000, lambda: update_plot(visualize_data_availability, inputname_site))
        time_series_window.mainloop()

    scale()
