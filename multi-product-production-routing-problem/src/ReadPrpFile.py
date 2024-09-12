    
class ReadPrpFile:
    def __init__(self,file_path):
        self.file_path=file_path

    def getDataSet(self):
        data = {}
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

            # Inicialização de variáveis
            section = None
            num_customers = 0
            num_products = 0
            num_vehicles = 0
            num_periods = 0
            demand = []
            i = 0

            while i < len(lines):
                line = lines[i].strip()

                # Identificação das seções
                if "Number of Customers" in line:
                    num_customers = int(line.split('=')[-1].strip())
                    data['num_customers'] = num_customers
                elif "Number of Products" in line:
                    num_products = int(line.split('=')[-1].strip())
                    data['num_products'] = num_products
                elif "Number of Vehicles" in line:
                    num_vehicles = int(line.split('=')[-1].strip())
                    data['num_vehicles'] = num_vehicles
                elif "Number of Periods" in line:
                    num_periods = int(line.split('=')[-1].strip())
                    data['num_periods'] = num_periods

                # Leitura de parâmetros
                elif line.startswith("B ="):
                    section = "B"
                    data['B'] = int(lines[i + 1].strip())
                    i += 1
                elif line.startswith("b_p ="):
                    section = "b_p"
                    data['b_p'] = list(map(int, lines[i + 1].strip().split()))
                    i += 1
                elif line.startswith("c_p ="):
                    section = "c_p"
                    data['c_p'] = list(map(int, lines[i + 1].strip().split()))
                    i += 1
                elif line.startswith("s_p ="):
                    section = "s_p"
                    data['s_p'] = list(map(int, lines[i + 1].strip().split()))
                    i += 1
                elif line.startswith("M ="):
                    section = "M"
                    data['M'] = int(lines[i + 1].strip())
                    i += 1
                elif line.startswith("U_pi ="):
                    section = "U_pi"
                    data['U_pi'] = []
                    for j in range(num_products):
                        data['U_pi'].append(list(map(int, lines[i + 1].strip().split())))
                        i += 1
                elif line.startswith("I_pi0 ="):
                    section = "I_pi0"
                    data['I_pi0'] = []
                    for j in range(num_products):
                        data['I_pi0'].append(list(map(int, lines[i + 1].strip().split())))
                        i += 1
                elif line.startswith("h_pi ="):
                    section = "h_pi"
                    data['h_pi'] = []
                    for j in range(num_products):
                        data['h_pi'].append(list(map(int, lines[i + 1].strip().split())))
                        i += 1
                elif line.startswith("C ="):
                    section = "C"
                    data['C'] = int(lines[i + 1].strip())
                    i += 1
                elif line.startswith("f ="):
                    section = "f"
                    data['f'] = int(lines[i + 1].strip())
                    i += 1
                elif line.startswith("a_ik ="):
                    section = "a_ik"
                    data['a_ik'] = []
                    for j in range(num_customers + 1):
                        data['a_ik'].append(list(map(int, lines[i + 1].strip().split())))
                        i += 1
                elif line.startswith("coordXY ="):
                    section = "coordXY"
                    data['coordXY'] = {"x": list(map(int, lines[i + 1].strip().split())),
                                    "y": list(map(int, lines[i + 2].strip().split()))}
                    i += 2
                elif line.startswith("d_pit ="):
                    section = "d_pit"
                    data['d_pit'] = []
                    for j in range(num_products * num_periods):
                        data['d_pit'].append(list(map(int, lines[i + 1].strip().split())))
                        i += 1

                i += 1
        return data