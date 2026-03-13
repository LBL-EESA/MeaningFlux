#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:50:49 2024

@author: Leila Hernadez Rodriguez 
Lawrence Berkeley National Laboratory

Description: Visualizes a Directional Contribution Rose by allowing selection of
wind direction, variables, and date ranges. The app bins data into 30° wind sectors
and plots the cumulative sum of the chosen variable per sector on a polar rose,
overlaid on a geospatial map with a 1 km buffer centered on the site. Negative
sector totals (e.g., FC/NEE) indicate net uptake (sink) from that direction.
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from tkinter import Tk, Frame, Label, StringVar, OptionMenu, Button, messagebox
from tkinter.ttk import Combobox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyproj import Transformer

rose_window = None  # renamed from hotspot_window
fig1 = None

def degrees_to_dms(deg):
    d = int(deg); m = int((deg - d) * 60); s = (deg - d - m/60) * 3600
    return f"{d}°{m}'{s:.2f}\""

def calc_plot_directional_contribution_rose(df, inputname_site, lat, lon):
    global rose_window, fig1

    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    def check_wind_direction_col():
        wd_columns = [col for col in df.columns if col.startswith('WD')]
        return wd_columns[0] if wd_columns else input("Enter the column label for wind direction: ")

    def check_variable_col():
        available = [c for c in df.columns if not c.startswith('WD') and c != 'TIMESTAMP_START']
        return available[0] if available else input("Enter the column label for the variable to plot: ")

    wd_col = check_wind_direction_col()
    var_col = check_variable_col()

    def get_and_plot():
        wind_direction_selected = wd_var_selection.get()
        variable_selected = variable_var_selection.get()

        start_date = pd.to_datetime(start_date_combobox.get())
        end_date   = pd.to_datetime(end_date_combobox.get())

        df_f = df[(df['TIMESTAMP_START'] >= start_date) & (df['TIMESTAMP_START'] <= end_date)].copy()

        df_f[wind_direction_selected] = df_f[wind_direction_selected].replace(['', -9999], np.nan)
        df_f[variable_selected]       = df_f[variable_selected].replace(['', -9999], np.nan)
        df_f.dropna(subset=[wind_direction_selected, variable_selected], inplace=True)

        if df_f.empty:
            messagebox.showwarning("Warning", "No valid data available for the selected range and variables")
            return

        wind_dir = df_f[wind_direction_selected]
        var_data = df_f[variable_selected]

        bins = np.arange(0, 361, 30)
        wind_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WNW']
        df_f['wind_bin'] = pd.cut(wind_dir, bins=bins, labels=wind_labels, right=False)

        sum_var_by_wind = df_f.groupby('wind_bin')[variable_selected].sum()

        gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        gdf_buffer = gdf.buffer(1000)  # 1 km

        cmap = plt.get_cmap('RdYlGn_r')
        norm_sum = mcolors.Normalize(vmin=sum_var_by_wind.min(), vmax=sum_var_by_wind.max())
        colors_sum = cmap(norm_sum(sum_var_by_wind))

        for widget in plot_frame.winfo_children():
            widget.destroy()

        global fig1
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        gdf_buffer.plot(ax=ax1, edgecolor='none', facecolor='none', linewidth=2)
        gdf.plot(ax=ax1, marker='o', color='red', markersize=100, label='Site Center')
        ctx.add_basemap(ax1, source=ctx.providers.Esri.WorldImagery, zoom=14)

        # Polar rose (directional contribution)
        ax_polar1 = fig1.add_axes([0.5 - 0.25, 0.5 - 0.25, 0.5, 0.5], projection='polar')
        angles = np.deg2rad(np.arange(0, 360, 30))
        ax_polar1.bar(angles, sum_var_by_wind.values, width=np.deg2rad(30),
                      color=colors_sum, alpha=0.6, align='edge')
        ax_polar1.set_theta_direction(-1)
        ax_polar1.set_theta_offset(np.pi / 2.0)
        ax_polar1.set_xticks(angles)
        ax_polar1.set_xticklabels(wind_labels)
        ax_polar1.patch.set_alpha(0.0)

        # Colorbar
        cbar_ax = fig1.add_axes([0.92, 0.25, 0.02, 0.5])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_sum); sm.set_array([])
        fig1.colorbar(sm, cax=cbar_ax, label=f'Sum of {variable_selected}')

        # Pretty lat/lon tick labels (DMS)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        x_ticks = ax1.get_xticks(); y_ticks = ax1.get_yticks()
        x_deg_ticks = [transformer.transform(x, gdf.geometry.y.iloc[0])[0] for x in x_ticks]
        y_deg_ticks = [transformer.transform(gdf.geometry.x.iloc[0], y)[1] for y in y_ticks]
        ax1.set_xticklabels([degrees_to_dms(t) for t in x_deg_ticks])
        ax1.set_yticklabels([degrees_to_dms(t) for t in y_deg_ticks])
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

        canvas = FigureCanvasTkAgg(fig1, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_plot():
        variable_selected = variable_var_selection.get()
        save_dir = 'Results/Directional_Contribution_Rose'
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{inputname_site}_directional_contribution_rose_{variable_selected}.png"
        filepath = os.path.join(save_dir, filename)
        fig1.savefig(filepath)
        print(f"Plot saved to {filepath}")

    def scale():
        global plot_frame, wd_var_selection, variable_var_selection, start_date_combobox, end_date_combobox, rose_window
        if rose_window is not None and rose_window.winfo_exists():
            rose_window.lift()
            return

        rose_window = Tk()
        rose_window.title('Directional Contribution Rose')
        rose_window.geometry("1000x700")

        left_frame = Frame(rose_window); left_frame.pack(side='left', fill='y')
        plot_frame = Frame(rose_window); plot_frame.pack(side='right', fill='both', expand=True)

        # Wind direction selector
        Label(left_frame, text="Select wind direction variable:").pack()
        wd_var_selection = StringVar(left_frame); wd_var_selection.set(wd_col)
        OptionMenu(left_frame, wd_var_selection, wd_col, *df.columns).pack()

        # Variable selector
        Label(left_frame, text="Select variable to plot:").pack()
        variable_var_selection = StringVar(left_frame); variable_var_selection.set(var_col)
        OptionMenu(left_frame, variable_var_selection, var_col, *df.columns).pack()

        # Date range
        Label(left_frame, text="Select start date:").pack()
        start_date_combobox = Combobox(left_frame, values=list(df['TIMESTAMP_START'].astype(str)))
        start_date_combobox.set(str(df['TIMESTAMP_START'].min())); start_date_combobox.pack()

        Label(left_frame, text="Select end date:").pack()
        end_date_combobox = Combobox(left_frame, values=list(df['TIMESTAMP_START'].astype(str)))
        end_date_combobox.set(str(df['TIMESTAMP_START'].max())); end_date_combobox.pack()

        def default_plot():
            try:
                wd_default = [c for c in df.columns if str(c).startswith('WD')][0]
            except IndexError:
                wd_default = wd_col
            wd_var_selection.set(wd_default)
            variable_var_selection.set(var_col if var_col in df.columns else df.columns[1])
            start_date_combobox.set(str(df['TIMESTAMP_START'].min()))
            end_date_combobox.set(str(df['TIMESTAMP_START'].max()))
            get_and_plot()

        Button(left_frame, text="Plot", command=get_and_plot).pack()
        Button(left_frame, text="Save Plot", command=save_plot).pack()

        Label(left_frame,
              text=("This rose shows the cumulative contribution of the selected variable by wind sector.\n"
                    "Bars represent 30° sectors (N … WNW). Negative totals (e.g., FC/NEE) indicate net uptake."))
     
        rose_window.after(100, default_plot)

        def on_closing():
            global rose_window
            rose_window.destroy()
            rose_window = None

        rose_window.protocol("WM_DELETE_WINDOW", on_closing)
        rose_window.mainloop()

    scale()

# Backward-compatible alias
def calc_plot_hotspots_direction(df, inputname_site, lat, lon):
    return calc_plot_directional_contribution_rose(df, inputname_site, lat, lon)
