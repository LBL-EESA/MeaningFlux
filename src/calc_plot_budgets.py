# -*- coding: utf-8 -*-
"""
Author: Leila Hernandez, LBNL
Date: August 5, 2024
Description: Plot annual budgets (annual time-integrated totals) for flux variables only.
             One bar per year; the optional % on top is the fraction of expected time steps
             that contain valid (non-missing) data for that year. Missing/-9999 do not count
             toward 100%.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, StringVar, Button, Frame, Label, OptionMenu, messagebox, BooleanVar, Checkbutton
from tkinter.ttk import Combobox  # kept (even if unused here)

# Define the units for the variables using LaTeX formatting (same as before)
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
    'SNO': r'$nmol \ \, NO \, mol^{-1}$',
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

annual_budget_window = None
show_percentage = None  # BooleanVar created in setup_interface()


def get_units(variable_name):
    key = variable_name.split('_1')[0]
    return variable_units.get(key, '')


def is_flux_variable(varname: str) -> bool:
    """
    Flux variables supported for annual budgets:
      1) Flux per area per time: * m^{-2} s^{-1}   (EC fluxes, photon fluxes, APAR/PPFD, SAP_FLOW)
      2) Energy flux per area:   W m^{-2}          (H, LE, G, NETRAD, SW_*, LW_*, R_UV*)
      3) Momentum/shear stress:  Kg m^{-1} s^{-2}  (TAU; equivalent to Pa = N m^{-2})
    """
    u = get_units(varname)
    if not u:
        return False

    u_low = u.lower()

    if ("m^{-2}" in u_low) and ("s^{-1}" in u_low):
        return True

    if ("w" in u_low) and ("m^{-2}" in u_low):
        return True

    if ("kg" in u_low) and ("m^{-1}" in u_low) and ("s^{-2}" in u_low):
        return True

    return False


def budget_ylabel(varname: str) -> str:
    """
    Y-axis label should represent an annual total (time-integrated):
      - For ... m^{-2} s^{-1}, an annual total is naturally expressed as ... m^{-2} yr^{-1}
      - For W m^{-2}, integrating over time gives J m^{-2} per year (energy)
      - TAU stays as-is (stress; if you later integrate, consider Pa·s or similar)
    This function updates the label only (does not change your calculation).
    """
    u = get_units(varname)
    if not u:
        return f"{varname}"

    u_low = u.lower()

    if ("m^{-2}" in u_low) and ("s^{-1}" in u_low):
        return f"{varname} ({u.replace('s^{-1}', 'yr^{-1}')})"

    if ("w" in u_low) and ("m^{-2}" in u_low):
        return f"{varname} ($J \\, m^{{-2}} \\, yr^{{-1}}$)"

    return f"{varname} ({u})"


def calc_plot_budgets(df, inputname_site):
    from tkinter import TclError

    global annual_budget_window, fig, show_percentage

    if df is None or df.empty:
        messagebox.showwarning("Warning", "Load the data first")
        return

    try:
        if annual_budget_window is not None and annual_budget_window.winfo_exists():
            annual_budget_window.lift()
            return
    except TclError:
        pass

    annual_budget_window = Tk()
    annual_budget_window.title('Annual Budget Plot')
    annual_budget_window.geometry("1000x700")

    # Convert TIMESTAMP_START to datetime (once)
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    # Create 'Year' once
    if 'Year' not in df.columns:
        df['Year'] = df['TIMESTAMP_START'].dt.year

    # Sampling interval (seconds)
    time_interval = (df['TIMESTAMP_START'].diff().dropna().mode()[0].total_seconds())
    expected_data_points_per_year = (365 * 24 * 3600) / time_interval

    # Restrict selection to flux variables only
    df_head = [col for col in sorted(df.columns) if is_flux_variable(col)]

    if not df_head:
        messagebox.showwarning(
            "No flux variables found",
            "Annual budgets are only available for flux variables "
            "(e.g., m^{-2} s^{-1}, W m^{-2}, or Kg m^{-1} s^{-2})."
        )
        return

    def get_and_plot():
        global fig

        userline = primary_var_selection.get()
        print(f"Selected variable: {userline}")

        if not is_flux_variable(userline):
            messagebox.showwarning(
                "Invalid selection",
                "Annual budgets are only available for flux variables "
                "(e.g., m^{-2} s^{-1}, W m^{-2}, or Kg m^{-1} s^{-2})."
            )
            return

        # Work on a slice (faster)
        df_work = df[['Year', userline]].copy()

        # Clean invalid values (-9999) and blanks
        df_work.loc[:, userline] = df_work[userline].replace(['', -9999, "-9999"], np.nan)

        data_percentage_per_year = []
        annual_sum = []

        conversion_factor_constant = 44.01  # keep your original approach

        for year, yearly_data in df_work.groupby('Year'):
            # Count ONLY valid values => -9999/NaN do not count toward 100%
            num_valid = yearly_data[userline].notna().sum()

            pct = (num_valid / expected_data_points_per_year) * 100
            data_percentage_per_year.append({'Year': year, 'Data_Percentage': pct})

            total_time_seconds = time_interval * num_valid

            if total_time_seconds > 0:
                conversion_factor = conversion_factor_constant / total_time_seconds
                yearly_sum = (yearly_data[userline].dropna() * conversion_factor).sum()
            else:
                yearly_sum = np.nan

            annual_sum.append({'Year': year, 'Annual_Sum': yearly_sum})

        data_percentage_df = pd.DataFrame(data_percentage_per_year)
        annual_sum_df = pd.DataFrame(annual_sum)

        #print("Data Percentage per Year:")
        #print(data_percentage_df)
        #print("\nAnnual Sum per Year:")
        #print(annual_sum_df)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(annual_sum_df['Year'], annual_sum_df['Annual_Sum'], color='skyblue')

        ax.set_title(f'{inputname_site} - Annual Budget for {userline}', fontsize=16)
        ax.set_xlabel('Year')
        ax.set_ylabel(budget_ylabel(userline))
        ax.grid(axis='y')

        ax.set_xticks(annual_sum_df['Year'])
        ax.set_xticklabels(annual_sum_df['Year'].astype(int), rotation=30)

        # Optional: annotate bars with % availability
        if show_percentage is not None and show_percentage.get():
            for bar, percentage in zip(bars, data_percentage_df['Data_Percentage']):
                height = bar.get_height()
                ax.annotate(f'{int(round(percentage))}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom')

        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def save_plot():
        global fig

        userline = primary_var_selection.get()

        # Ensure folder exists (THIS fixes your save issue)
        save_dir = os.path.join('Results', 'Annual_budget_plots')
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{inputname_site.split('.')[0]}_{userline}_annual_budget.png"
        filepath = os.path.join(save_dir, filename)

        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {filepath}")

    def setup_interface():
        global plot_frame, primary_var_selection, annual_budget_window, show_percentage

        left_frame = Frame(annual_budget_window)
        left_frame.pack(side='left', fill='y')

        plot_frame = Frame(annual_budget_window)
        plot_frame.pack(side='right', fill='both', expand=True)

        Label(left_frame, text="Select flux variable:").pack()

        # Prefer FC if available; else first flux variable
        default_var = next((col for col in df_head if col.startswith('FC')), df_head[0])

        primary_var_selection = StringVar(left_frame)
        primary_var_selection.set(default_var)

        primary_var_menu = OptionMenu(left_frame, primary_var_selection, *df_head)
        primary_var_menu.pack()

        selectbutton = Button(left_frame, text="Plot", command=get_and_plot)
        selectbutton.pack()

        # Toggle to show/hide percentages
        show_percentage = BooleanVar()
        show_percentage.set(True)

        percentage_checkbox = Checkbutton(
            left_frame,
            text="Show data availability (%)",
            variable=show_percentage,
            command=get_and_plot
        )
        percentage_checkbox.pack()

        savebutton = Button(left_frame, text="Save Plot", command=save_plot)
        savebutton.pack()

        note_text = (
            "Bars: annual time-integrated total.\n"
            "Top %: valid data coverage (excludes -9999)."
        )
        note_label = Label(left_frame, text=note_text, justify="left", wraplength=240)
        note_label.pack(pady=(10, 0), anchor="w")

        # Plot immediately on open
        annual_budget_window.after(5, get_and_plot)

        annual_budget_window.mainloop()

    setup_interface()