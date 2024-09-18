# graph_display.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class GraphDisplay:
    def __init__(self, parent):
        self.ploat = True
        self.num_graphs_per_page = 10  # Limite de gráficos por página
        self.num_tabs = 3  # Número de abas
        self.num_graphs = 12  # Total de gráficos a serem distribuídos nas abas
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(expand=1, fill="both")
        self.graph_widgets = []

        # Armazena dados de cada aba para a paginação
        self.tabs_data = {}

        # Criar as abas
        self.create_tabs()

    def create_tabs(self):
        """Cria múltiplas abas e adiciona paginação dentro de cada aba."""
        for i in range(self.num_tabs):
            tab_frame = ttk.Frame(self.tabs)
            self.tabs.add(tab_frame, text=f"Tab {i + 1}")

            # Adiciona paginação de gráficos dentro da aba
            self.create_graph_pagination(tab_frame, i)

    def create_graph_pagination(self, parent, tab_index):
        """Cria a paginação de gráficos dentro de cada aba."""
        # Cria a estrutura da aba com scrollbar e canvas
        canvas = tk.Canvas(parent)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar_y = tk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

        scrollbar_x = tk.Scrollbar(parent, orient=tk.HORIZONTAL, command=canvas.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        canvas_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=canvas_frame, anchor="nw")

        canvas_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Frame para os gráficos e botões de paginação
        pagination_frame = tk.Frame(canvas_frame)
        pagination_frame.pack(fill=tk.X)

        # Botões de navegação
        prev_button = tk.Button(pagination_frame, text="Previous", command=lambda: self.previous_page(tab_index), state=tk.DISABLED)
        prev_button.pack(side=tk.LEFT)

        page_label = tk.Label(pagination_frame, text="Page 1")
        page_label.pack(side=tk.LEFT, padx=20)

        next_button = tk.Button(pagination_frame, text="Next", command=lambda: self.next_page(tab_index))
        next_button.pack(side=tk.LEFT)

        # Frame para os gráficos
        graph_frame = tk.Frame(canvas_frame)
        graph_frame.pack(fill=tk.BOTH, expand=1)

        # Armazena o estado de cada aba (gráficos, página atual)
        self.tabs_data[tab_index] = {
            "parent": parent,
            "graph_frame": graph_frame,
            "page_label": page_label,
            "prev_button": prev_button,
            "next_button": next_button,
            "current_page": 0,
            "total_graphs": self.num_graphs,
        }

        # Exibe os gráficos iniciais
        self.display_graphs(tab_index)

    def create_line_chart(self, parent, graph_number):
        """Cria um gráfico de linha usando matplotlib."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [2 * graph_number, 4 * graph_number, 6 * graph_number, 8 * graph_number, 10 * graph_number]

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x_data, y_data, label=f"Line Plot {graph_number}", marker="o")
        ax.set_title(f"Line Chart {graph_number}")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.graph_widgets.append(canvas_widget)
        self.ploat = False

    def destroy_graphs(self):
        """Destroy all the graph widgets from the tabs."""
        for widget in self.graph_widgets:
            widget.destroy()  # Remove cada gráfico da interface
        self.graph_widgets.clear()  # Limpa a lista de widgets

        # Destrói o notebook que contém as abas
        self.tabs.destroy()
    
    def getPloat(self):
        print(self.ploat)
        return self.ploat

    def display_graphs(self, tab_index):
        """Exibe os gráficos correspondentes à página atual."""
        tab_data = self.tabs_data[tab_index]
        graph_frame = tab_data["graph_frame"]
        
        # Limpa os gráficos anteriores
        for widget in graph_frame.winfo_children():
            widget.destroy()

        start_graph = tab_data["current_page"] * self.num_graphs_per_page
        end_graph = min(start_graph + self.num_graphs_per_page, tab_data["total_graphs"])

        for graph_number in range(start_graph, end_graph):
            self.create_line_chart(graph_frame, graph_number + 1)

        # Atualiza o estado dos botões de navegação
        self.update_pagination_controls(tab_index)

    def update_pagination_controls(self, tab_index):
        """Atualiza os controles de paginação para a aba atual."""
        tab_data = self.tabs_data[tab_index]
        current_page = tab_data["current_page"]
        total_pages = (tab_data["total_graphs"] + self.num_graphs_per_page - 1) // self.num_graphs_per_page

        tab_data["page_label"].config(text=f"Page {current_page + 1}")

        if current_page == 0:
            tab_data["prev_button"].config(state=tk.DISABLED)
        else:
            tab_data["prev_button"].config(state=tk.NORMAL)

        if current_page == total_pages - 1:
            tab_data["next_button"].config(state=tk.DISABLED)
        else:
            tab_data["next_button"].config(state=tk.NORMAL)

    def next_page(self, tab_index):
        """Navega para a próxima página de gráficos."""
        tab_data = self.tabs_data[tab_index]
        if (tab_data["current_page"] + 1) * self.num_graphs_per_page < tab_data["total_graphs"]:
            tab_data["current_page"] += 1
            self.display_graphs(tab_index)

    def previous_page(self, tab_index):
        """Navega para a página anterior de gráficos."""
        tab_data = self.tabs_data[tab_index]
        if tab_data["current_page"] > 0:
            tab_data["current_page"] -= 1
            self.display_graphs(tab_index)