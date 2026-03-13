#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:05:10 2024

Open the BADM window. Returns the updated BADM values included by the user

@author: Leila Hernadez Rodriguez 
Lawrence Berkeley National Laboratory
"""    


from tkinter import *
from tkinter import Toplevel
import webbrowser
from folium import Map, Marker
from folium.plugins import MiniMap
import tkhtmlview  # Import the tkhtmlview module
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from folium import plugins

class BADMWindow:
    def __init__(self, master):
        self.master = master
        self.setup_window()
        self.set_default_values()  # Call the method to initialize default values

    def open_FLUXNET_BADM_url(self, event):
        webbrowser.open("https://fluxnet.org/badm-data-product/")

    def open_AMERIFLUX_BADM_url(self, event):
        webbrowser.open("https://ameriflux.lbl.gov/data/badm/")

    def setup_window(self):
        self.master.title('Input BADM')

        # Main frame to hold input fields
        self.main_frame = Frame(self.master)
        self.main_frame.pack(padx=10, pady=10, side=LEFT)

        # Frame for input fields
        self.branch = LabelFrame(self.main_frame, text="BADM (Biological, Ancillary, Disturbance and Metadata)")
        self.branch.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        row_count = 0
        first_column = 0
        fourth_column = 3
        entry_width = 10

        # Label for additional text
        additional_text = ("Replace the reference values with the BADM of your field site.\n"
                           "After updating each value, click on another cell to save it.\n"
                           "The updated values will show up in the console window.")
        
        additional_label = Label(self.branch, text=additional_text, wraplength=500)  # Adjust wraplength as needed
        additional_label.grid(row=row_count, column=first_column, columnspan=2, pady=5)
        row_count += 1
        
        # Create a link to FLUXNET BADM website
        link_label = Label(self.branch, text="Visit FLUXNET BADM Data Product website", fg="blue", cursor="hand2")
        link_label.grid(row=row_count, column=first_column, columnspan=2, pady=5)
        link_label.bind("<Button-1>", self.open_FLUXNET_BADM_url)
        row_count += 1

        # Create a link to AMERIFLUX BADM website
        link_label = Label(self.branch, text="Visit AMERIFLUX BADM Data Product website", fg="blue", cursor="hand2")
        link_label.grid(row=row_count, column=first_column, columnspan=2, pady=5)
        link_label.bind("<Button-1>", self.open_AMERIFLUX_BADM_url)
        row_count += 1

        self.lat = DoubleVar()
        self.lon = DoubleVar()
        self.altitude = DoubleVar()
        self.UTC_offset = DoubleVar()
        self.z = DoubleVar()
        self.h_canopy_max = DoubleVar()
        self.bhl = DoubleVar()

        # Eddy covariance tower latitude [m]
        h0 = Label(self.branch, text='Latitude (decimal degrees)')
        h0.grid(row=row_count, column=first_column)
        self.lat_entry = Entry(self.branch, textvariable=self.lat, width=entry_width)
        self.lat_entry.grid(row=row_count, column=fourth_column)
        self.lat_entry.bind("<FocusOut>", self.update_lat)  # Bind the callback on focus out
        row_count += 1

        # Eddy covariance tower longitude [m]
        h0 = Label(self.branch, text='Longitude (decimal degrees)')
        h0.grid(row=row_count, column=first_column)
        self.lon_entry = Entry(self.branch, textvariable=self.lon, width=entry_width)
        self.lon_entry.grid(row=row_count, column=fourth_column)
        self.lon_entry.bind("<FocusOut>", self.update_lon)  # Bind the callback on focus out
        row_count += 1

        # Altitude [m]
        h0 = Label(self.branch, text='Altitude (m)')
        h0.grid(row=row_count, column=first_column)
        self.altitude_entry = Entry(self.branch, textvariable=self.altitude, width=entry_width)
        self.altitude_entry.grid(row=row_count, column=fourth_column)
        self.altitude_entry.bind("<FocusOut>", self.update_altitude)  # Bind the callback on focus out
        row_count += 1

        # UTC offset [hours]
        h0 = Label(self.branch, text='UTC offset (hours)')
        h0.grid(row=row_count, column=first_column)
        self.utc_offset_entry = Entry(self.branch, textvariable=self.UTC_offset, width=entry_width)
        self.utc_offset_entry.grid(row=row_count, column=fourth_column)
        self.utc_offset_entry.bind("<FocusOut>", self.update_utc_offset)  # Bind the callback on focus out
        row_count += 1

        # z - Eddy covariance measurement height [m]
        h0 = Label(self.branch, text='z = EC measurement height (m)')
        h0.grid(row=row_count, column=first_column)
        self.z_entry = Entry(self.branch, textvariable=self.z, width=entry_width)
        self.z_entry.grid(row=row_count, column=fourth_column)
        self.z_entry.bind("<FocusOut>", self.update_z)  # Bind the callback on focus out
        row_count += 1

        # h_canopy_max - Maximum canopy height [m]
        h0 = Label(self.branch, text='h_canopy_max = Maximum canopy height (m)')
        h0.grid(row=row_count, column=first_column)
        self.h_canopy_max_entry = Entry(self.branch, textvariable=self.h_canopy_max, width=entry_width)
        self.h_canopy_max_entry.grid(row=row_count, column=fourth_column)
        self.h_canopy_max_entry.bind("<FocusOut>", self.update_h_canopy_max)  # Bind the callback on focus out
        row_count += 1

        # # bhl - Boundary Layer Height [m]
        # h0 = Label(self.branch, text='bhl = Boundary layer height (m)')
        # h0.grid(row=row_count, column=first_column)
        # self.bhl_entry = Entry(self.branch, textvariable=self.bhl, width=entry_width)
        # self.bhl_entry.grid(row=row_count, column=fourth_column)
        # self.bhl_entry.bind("<FocusOut>", self.update_bhl)  # Bind the callback on focus out
        # row_count += 1        
        
        # Create a button to open the map
        map_button = Button(self.branch, text="Open location in Google Maps", command=self.open_map)
        map_button.grid(row=row_count, column=first_column, columnspan=2, pady=5)
        row_count += 1
               
        
    def open_map(self):
        latitude = UpdatedValues.lat
        longitude = UpdatedValues.lon

        # Construct the Google Maps URL with the specified latitude and longitude
        google_maps_url = f"https://www.google.com/maps/place/{latitude},{longitude}"

        # Open the URL in the default web browser
        webbrowser.open(google_maps_url)

    def set_default_values(self):
        # Set or update default values here
        self.lat.set(UpdatedValues.lat)
        self.lon.set(UpdatedValues.lon)
        self.altitude.set(UpdatedValues.altitude)
        self.UTC_offset.set(UpdatedValues.UTC_offset)
        self.z.set(UpdatedValues.z)
        self.h_canopy_max.set(UpdatedValues.h_canopy_max)
        #self.bhl.set(UpdatedValues.bhl)

        # Update Entry widgets with the default or updated values
        self.lat_entry.delete(0, END)
        if self.lat.get() != "":
            self.lat_entry.insert(0, self.lat.get())

        self.lon_entry.delete(0, END)
        if self.lon.get() != "":
            self.lon_entry.insert(0, self.lon.get())

        self.altitude_entry.delete(0, END)
        if self.altitude.get() != "":
            self.altitude_entry.insert(0, self.altitude.get())

        self.utc_offset_entry.delete(0, END)
        if self.UTC_offset.get() != "":
            self.utc_offset_entry.insert(0, self.UTC_offset.get())

        self.z_entry.delete(0, END)
        if self.z.get() != "":
            self.z_entry.insert(0, self.z.get())

        self.h_canopy_max_entry.delete(0, END)
        if self.h_canopy_max.get() != "":
            self.h_canopy_max_entry.insert(0, self.h_canopy_max.get())

        # self.bhl_entry.delete(0, END)
        # if self.bhl.get() != "":
        #     self.bhl_entry.insert(0, self.bhl.get())

    def update_lat(self, event):
        new_value = self.lat_entry.get()
        if new_value != "":
            new_value = float(new_value)
            self.lat.set(new_value)
            UpdatedValues.lat = new_value
            print(f"Updated lat: {new_value}")

    def update_lon(self, event):
        new_value = self.lon_entry.get()
        if new_value != "":
            new_value = float(new_value)
            self.lon.set(new_value)
            UpdatedValues.lon = new_value
            print(f"Updated lon: {new_value}")

    def update_altitude(self, event):
        new_value = self.altitude_entry.get()
        if new_value != "":
            new_value = float(new_value)
            self.altitude.set(new_value)
            UpdatedValues.altitude = new_value
            print(f"Updated altitude: {new_value}")

    def update_utc_offset(self, event):
        new_value = self.utc_offset_entry.get()
        if new_value != "":
            new_value = float(new_value)
            self.UTC_offset.set(new_value)
            UpdatedValues.UTC_offset = new_value
            print(f"Updated UTC_offset: {new_value}")

    def update_z(self, event):
        new_value = self.z_entry.get()
        if new_value != "":
            new_value = float(new_value)
            self.z.set(new_value)
            UpdatedValues.z = new_value
            print(f"Updated z: {new_value}")

    def update_h_canopy_max(self, event):
        new_value = self.h_canopy_max_entry.get()
        if new_value != "":
            new_value = float(new_value)
            self.h_canopy_max.set(new_value)
            UpdatedValues.h_canopy_max = new_value
            print(f"Updated h_canopy_max: {new_value}")

    # def update_bhl(self, event):
    #     new_value = self.bhl_entry.get()
    #     if new_value != "":
    #         new_value = float(new_value)
    #         self.bhl.set(new_value)
    #         UpdatedValues.bhl = new_value
    #         print(f"Boundary layer height: {new_value}")

    def on_window_close(self):
        self.master.destroy()

class UpdatedValues:
    lat = 34.4121
    lon = -91.6752
    altitude = 10
    UTC_offset = -5
    z = 4.6
    h_canopy_max = 3.5
    #bhl = 700
    
def open_BADM_window():
    root = Tk()
    badm_window = BADMWindow(root)
    root.mainloop()
    return UpdatedValues

if __name__ == "__main__":
    root = Tk()
    badm_window = BADMWindow(root)
    root.mainloop()
