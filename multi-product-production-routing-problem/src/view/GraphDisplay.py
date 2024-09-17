# graph_display.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GraphDisplay:
    def __init__(self, parent):
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(expand=1, fill="both")

    def create_graph_tabs(self, num_tabs):
        """Create multiple tabs, each with a line chart."""
        for i in range(num_tabs):
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text=f"Graph {i+1}")

            # Create and display a line chart inside each tab
            self.create_line_chart(tab)

    def create_line_chart(self, parent):
        """Create a sample line chart using matplotlib."""
        # Example data for the line chart
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 6, 8, 10]

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_data, y_data, label="Line Plot", marker="o")
        ax.set_title("Line Chart Example")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()

        # Embed the plot in the tkinter tab
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
