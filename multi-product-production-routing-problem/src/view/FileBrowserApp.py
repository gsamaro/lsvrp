# main.py
import tkinter as tk
from tkinter import ttk
from src.view.FileBrowser import FileBrowser
from src.view.GraphDisplay import GraphDisplay
import sys

class FileBrowserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Directory File List with Line Graphs")

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
        self.file_browser = FileBrowser(self.left_frame, self.on_file_selected)

        # Graph display setup
        self.graph_display = GraphDisplay(self.right_frame)
        self.graph_display.create_graph_tabs(3)  # Create 3 tabs with graphs as an example

    def on_file_selected(self, file_path):
        """Callback function called when a file is selected."""
        print(f"File selected: {file_path}")

    def on_close(self):
        """Handler to ensure Python exits when the window is closed."""
        print("Window closed, terminating the process.")
        self.root.quit()  # Properly close the tkinter main loop
        self.root.destroy()  # Destroy the tkinter window
        sys.exit(0)  # Exit the Python process