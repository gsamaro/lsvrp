from src.view.FileBrowserApp import FileBrowserApp as FBA
import tkinter as tk

# Initialize the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = FBA(root)
    app.listenter()
    root.mainloop()


# from src.ReadPrpFile import ReadPrpFile as RD
# from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP

# if __name__ == "__main__":
#     data = RD("./data/DATA_PRP_5C/PRP1_C5_P3_V1_T6_S1.dat")

#     mpp = MPPRP(data.getDataSet())
#     data.toString()
#     mpp.solver()
    