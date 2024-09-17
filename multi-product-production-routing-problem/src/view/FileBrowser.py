# file_browser.py
import os
import threading
import tkinter as tk
from tkinter import filedialog, Listbox, Scrollbar
from src.ReadPrpFile import ReadPrpFile as RD
from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP


class FileBrowser:
    def __init__(self, parent, on_file_selected):
        self.parent = parent
        self.on_file_selected = on_file_selected  # Callback to notify when a file is selected
        self.full_files_list = []
        self.initial_directory = "./data"  # Replace with your desired path

        # Creating the interface components
        self.list_button = tk.Button(parent, text="Select Directory", command=self.list_files)
        self.list_button.pack(pady=10)

        self.directory_label = tk.Label(parent, text="No directory selected", wraplength=300, justify="left")
        self.directory_label.pack(pady=10)

        self.search_entry = tk.Entry(parent, width=50)
        self.search_entry.pack(pady=10)
        self.search_entry.bind("<KeyRelease>", self.filter_files)

        # Create a scrollbar for the listbox
        self.scrollbar = Scrollbar(parent)

        # Listbox to display the files in the directory
        self.listbox = Listbox(parent, width=50, height=15, yscrollcommand=self.scrollbar.set)
        self.listbox.pack(pady=10)

        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Button to execute the process with the selected file
        self.open_button = tk.Button(parent, text="Open Selected File", command=self.run_process)
        self.open_button.pack(pady=10)

        # Fields for displaying file information
        self.file = tk.Label(parent, text="Data set: ")
        self.file.pack(pady=5)
        self.file_name_number_customer = tk.Label(parent, text="Number of Customers: ")
        self.file_name_number_customer.pack(pady=5)
        self.file_name_number_products = tk.Label(parent, text="Number of Products: ")
        self.file_name_number_products.pack(pady=5)
        self.file_name_number_vehicles = tk.Label(parent, text="Number of Vehicles: ")
        self.file_name_number_vehicles.pack(pady=5)
        self.file_name_number_preriods = tk.Label(parent, text="Number of Periods: ")
        self.file_name_number_preriods.pack(pady=5)

        # Label for showing the status
        self.status_label = tk.Label(parent, text="", wraplength=300, justify="left")
        self.status_label.pack(pady=10)

    def list_files(self):
        directory = filedialog.askdirectory(initialdir=self.initial_directory)

        if directory:
            self.full_files_list = os.listdir(directory)
            self.listbox.delete(0, tk.END)

            for file in self.full_files_list:
                self.listbox.insert(tk.END, os.path.basename(file))

            self.directory_label['text'] = f"Selected directory: {directory}"

    def filter_files(self, event=None):
        term = self.search_entry.get().lower()
        self.listbox.delete(0, tk.END)

        for file in self.full_files_list:
            if term in os.path.basename(file).lower():
                self.listbox.insert(tk.END, os.path.basename(file))

    def run_process(self):
        selected = self.listbox.curselection()

        if selected:
            file_name = self.listbox.get(selected)
            directory = self.directory_label['text'].replace("Selected directory: ", "")
            file_path = os.path.join(directory, file_name)

            # Create a thread to run the process in the background
            thread = threading.Thread(target=self.call_process, args=(file_path,))
            thread.start()

    # Function that executes the process (opens the file)
    def call_process(self, file_path):
        read = RD(file_path)
        dataSet = read.getDataSet()

        # Update file information fields
        self.display_file_info(dataSet,file_path)

        mpprp = MPPRP(dataSet).solver()

    # Function to display file information
    def display_file_info(self, dataSet, file_path):
        file_name = os.path.basename(file_path)

        # Update the labels with file information
        self.file['text'] = f"Data set:  {file_name}"
        self.file_name_number_customer['text'] = f"Number of Customers: {dataSet['num_customers']}"
        self.file_name_number_products['text'] = f"Number of Products:  {dataSet['num_products']}"
        self.file_name_number_vehicles['text'] = f"Number of Vehicles: {dataSet['num_vehicles']}"
        self.file_name_number_preriods['text'] = f"Number of Periods: {dataSet['num_periods']}"

