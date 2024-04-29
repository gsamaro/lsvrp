import tkinter as tk

class Drawer:
    def __init__(self):

        self.clients = ""
        self.instance = "instance"
        self.validete = ""
        self.window = tk.Tk()
        self.window.title("Input Interface")

        # Labels and input fields
        tk.Label(self.window, text="Clients:").grid(row=0, column=0, padx=5, pady=5)
        self.input1 = tk.Entry(self.window)
        self.input1.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self.window, text="Instance:").grid(row=1, column=0, padx=5, pady=5)
        self.input2 = tk.Entry(self.window)
        self.input2.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self.window, text="Validar Imput:").grid(row=2, column=0, padx=5, pady=5)
        self.input3 = tk.Entry(self.window)
        self.input3.grid(row=2, column=1, padx=5, pady=5)

        # Button to submit values
        self.button_submit = tk.Button(self.window, text="Submit", command=self.submit_values)
        self.button_submit.grid(row=3, columnspan=2, padx=5, pady=5)

    def submit_values(self):
        self.clients = self.input1.get()
        self.instance = self.instance+self.input2.get()
        self.validete = self.input3.get()
        self.window.destroy()

    def getInformations(self):
        return{
            'clients': self.clients,
            'instance':self.instance,
            'validete':self.validete
        }
    def mainloop(self):
        self.window.mainloop()