# main.py
import tkinter as tk
from tkinter import ttk
import threading
from src.view.FileBrowser import FileBrowser
from src.view.GraphDisplay import GraphDisplay
import sys
import time

class FileBrowserApp:
    def __init__(self, root):
        self.thread = 0
        self.listener = True
        self.ploatGW=False
        self.button=False
        self.root = root
        self.graph_display = 0
        self.root.title("Optimization screen")
        # Capture the screen width and height
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set the window size dynamically to 80% of the screen dimensions
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        # Center the window on the screen
        position_right = int(screen_width/2 - window_width/2)
        position_down = int(screen_height/2 - window_height/2)

        # Set the geometry of the window (widthxheight+Xposition+Yposition)
        self.root.geometry(f"{window_width}x{window_height}+{position_right}+{position_down}")

        # Capture the window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Create a PanedWindow to divide the layout
        self.paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=1)

        # Left frame for the file list
        self.left_frame = tk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, minsize=200)

        # Right frame for the tabs with graphs
        self.right_frame = tk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, minsize=400)

        # File browser setup
        self.file_browser = FileBrowser(self.left_frame, self.on_file_selected, self.ploatGr)

        # Graph display setup
        # self.graph_display.create_graph_tabs(10)  # Create 3 tabs with graphs as an example

    def on_file_selected(self, file_path):
        """Callback function called when a file is selected."""
        if(self.button==True):
            self.graph_display.destroy_graphs()
            self.button=False
            self.ploatGW=True
        print(f"File selected: {file_path}")

    def on_close(self):
        """Handler to ensure Python exits when the window is closed."""
        print("Window closed, terminating the process.")
        self.file_browser.endProcess()
        self.listener = False
        self.thread.join()
        self.root.quit()  # Properly close the tkinter main loop
        self.root.destroy()  # Destroy the tkinter window
        sys.exit(0)  # Exit the Python process
    
    def listenter(self):
        self.thread = threading.Thread(target=self.call_process,args=("tread",))
        self.thread.start()
     
    def ploatGr(self):
        if(self.ploatGW==True):
            self.graph_display = GraphDisplay(self.right_frame)
            self.button=True

    def call_process(self,string):
        
        ploat = True
        while(self.listener):
            process = self.file_browser.getResults()
            if(process['end']==True and ploat==True): 
                self.ploatGW = True

                if(self.button==True):
                    ploat = self.graph_display.getPloat()
            
            time.sleep(0.5) 
            
            
       
