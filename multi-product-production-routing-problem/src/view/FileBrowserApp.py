import tkinter as tk
from tkinter import ttk
from src.view.FileBrowser import FileBrowser
from src.view.GraphDisplay import GraphDisplay

class FileBrowserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Directory File List with Line Graphs")

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