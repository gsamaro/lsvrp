# file_browser.py
import os
import threading
import tkinter as tk
from tkinter import filedialog, Listbox, Scrollbar
from src.ReadPrpFile import ReadPrpFile as RD
from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.Converter import toStopPoint, inveterMatrix
import time

class FileBrowser:
    def __init__(self, parent, on_file_selected, on_plot_results):

        self.threads = []
        self.results = {
            'end': False,
        }
        self.parent = parent
        self.on_plot_results = on_plot_results
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

        # Create a frame to hold the Listbox and Scrollbar
        self.list_frame = tk.Frame(parent)
        self.list_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Create a scrollbar for the listbox and place it to the right of the Listbox
        self.scrollbar = Scrollbar(self.list_frame)

        # Listbox to display the files in the directory
        self.listbox = Listbox(self.list_frame, width=50, height=15, yscrollcommand=self.scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Button to execute the process with the selected file
        self.open_button = tk.Button(parent, text="Open Selected File", command=self.run_process)
        self.open_button.pack(pady=10)

        # Fields for displaying file information
        self.file = tk.Label(parent, text="Data set: ")
        self.file.pack(pady=5)
        self.file_info = tk.Label(parent, text="Customers: , Products: , \nVehicles: , Periods: ")
        self.file_info.pack(pady=5)
        self.file_rest = tk.Label(parent, text="==")
        self.file_rest.pack(pady=5)


        # Button to execute the process with the selected file
        self.open_button = tk.Button(parent, text="Ploat Results", command=self.on_plot_results)
        self.open_button.pack(pady=10)

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

    def call_process(self, file_path):
        self.on_file_selected("Nova otimizaçao")
        self.results = {
            'end': False,
        }

        read = RD(file_path)
        dataSet = read.getDataSet()

        # Update file information fields
        self.display_file_info(dataSet,file_path)

        init = time.time()
        mpprp = MPPRP(dataSet)
        self.threads.append(mpprp)
        mpprp.solver()
        Z,X,Y,I,R,Q,FO,GAP = mpprp.getResults()

        periods = []
        for t in range(len(Z)):
            veicule = []
            for v in range(len(Z[t])):
                veicule.append(toStopPoint(Z[t][v]))
            periods.append(veicule)
        fim = time.time()
        self.display_file_resutl(FO,GAP,round(fim-init))
        self.results = {
            'end': True,
            'periods': periods
        }

    # Function to display file information
    def display_file_info(self, dataSet, file_path):
        file_name = os.path.basename(file_path)

        # Update the labels with file information
        self.file['text'] = f"Data set:  {file_name}"
        self.file_info['text'] = f"Customers: {dataSet['num_customers']}, Products:{dataSet['num_products']}, \nVehicles:{dataSet['num_vehicles']}, Periods:{dataSet['num_periods']}."
        self.file_rest['text'] = "=="

    def display_file_resutl(self,fo,gap,time):
        self.file_rest['text'] = f"FO: {fo}, GAP:{gap}, time:{time}"

    def endProcess(self):
        for i in range(len(self.threads)):
            self.threads[i].terminate()

    def getResults(self):
        return self.results