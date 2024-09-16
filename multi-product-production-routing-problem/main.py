
import os
import tkinter as tk
from tkinter import filedialog, Listbox, Scrollbar
from tkinter import ttk
import subprocess
import threading
import time  # Simular tempo para a barra de progresso
from src.ReadPrpFile import ReadPrpFile as RD
from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP

#chama a classe de otimização
# def process(file_path):
#     read = RD(file_path)
#     mpprp = MPPRP(read.getDataSet()).solver()
#     return "ok"

# Caminho específico onde a janela de diálogo será aberta
diretorio_inicial = "./data"

# Variável global para armazenar a lista completa de arquivos
arquivos_completos = []

# Função para listar os arquivos de um diretório selecionado
def listar_arquivos():
    global arquivos_completos
    # Abrir uma janela de diálogo para selecionar um diretório, começando no caminho especificado
    diretorio = filedialog.askdirectory(initialdir=diretorio_inicial)

    if diretorio:
        # Limpar a lista de arquivos antes de listar os novos
        listbox.delete(0, tk.END)
        
        # Listar todos os arquivos e pastas no diretório selecionado
        arquivos_completos = os.listdir(diretorio)

        # Adicionar apenas os nomes dos arquivos à listbox
        for arquivo in arquivos_completos:
            listbox.insert(tk.END, os.path.basename(arquivo))  # Apenas o nome do arquivo

        # Exibir o caminho do diretório
        label_diretorio['text'] = f"Diretório selecionado: {diretorio}"

# Função para aplicar o filtro de pesquisa
def filtrar_arquivos(event=None):
    termo = entrada_pesquisa.get().lower()  # Obter o termo da caixa de texto e converter para minúsculas
    listbox.delete(0, tk.END)  # Limpar a listbox
    for arquivo in arquivos_completos:
        if termo in os.path.basename(arquivo).lower():  # Filtrar pelo nome do arquivo em minúsculas
            listbox.insert(tk.END, os.path.basename(arquivo))  # Exibir apenas os arquivos que correspondem ao filtro

# Função que chama um processo para abrir o arquivo selecionado
def executar_processo():
    # Obter o arquivo selecionado da listbox
    selecionado = listbox.curselection()
    
    if selecionado:
        nome_arquivo = listbox.get(selecionado)  # Obter apenas o nome do arquivo
        caminho_arquivo = os.path.join(label_diretorio['text'].replace("Diretório selecionado: ", ""), nome_arquivo)  # Construir o caminho completo

        # Criar um thread para executar o processo em segundo plano
        thread = threading.Thread(target=chamar_processo, args=(caminho_arquivo,))
        thread.start()

# Função que executa o processo (abrir o arquivo)
def chamar_processo(caminho_arquivo):
    read = RD(caminho_arquivo)
    mpprp = MPPRP(read.getDataSet()).solver()

# Criação da interface gráfica
root = tk.Tk()
root.title("Lista de Arquivos do Diretório")

# Botão para selecionar e listar os arquivos
botao_listar = tk.Button(root, text="Selecionar Diretório", command=listar_arquivos)
botao_listar.pack(pady=10)

# Label para mostrar o diretório selecionado
label_diretorio = tk.Label(root, text="Nenhum diretório selecionado", wraplength=300, justify="left")
label_diretorio.pack(pady=10)

# Entrada para pesquisa de arquivos
entrada_pesquisa = tk.Entry(root, width=50)
entrada_pesquisa.pack(pady=10)
entrada_pesquisa.bind("<KeyRelease>", filtrar_arquivos)  # Filtrar à medida que o usuário digita

# Criar um scrollbar para a listbox
scrollbar = Scrollbar(root)

# Listbox para exibir os arquivos do diretório
listbox = Listbox(root, width=50, height=15, yscrollcommand=scrollbar.set)
listbox.pack(pady=10)

# Configurar o scrollbar para funcionar com a listbox
scrollbar.config(command=listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Botão para executar o processo com o arquivo selecionado
botao_executar = tk.Button(root, text="Abrir Arquivo Selecionado", command=executar_processo)
botao_executar.pack(pady=10)

# Label para mostrar o status
label_status = tk.Label(root, text="", wraplength=300, justify="left")
label_status.pack(pady=10)

# Iniciar o loop da interface gráfica
root.mainloop()