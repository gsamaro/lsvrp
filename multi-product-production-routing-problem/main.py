from src.view.FileBrowserApp import FileBrowserApp as FBA
import tkinter as tk

# Initialize the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = FBA(root)
    app.listenter()
    root.mainloop()