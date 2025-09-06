import pandas as pd
import json
from tabulate import tabulate

class TablesResult:

    def process(datas=None):
        if datas is None:
            datas = []

        def get_instance_class(instance_number):
            if 1 <= instance_number <= 10:
                return "I"
            elif 11 <= instance_number <= 20:
                return "II"
            elif 21 <= instance_number <= 30:
                return "III"
            elif 31 <= instance_number <= 40:
                return "IV"
            else:
                return "Unknown"

        results = []

        for data in datas:
            directory = data['data']
            results_file = []

            for file in data['files']:
                name_file = file.replace('.dat', '')

                with open(f'./out/{directory}/{name_file}/result.json', 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                params = name_file.split("_")
                instance_number = int(params[0].replace('PRP', ''))
                number_clients = int(params[1].replace('C', ''))
                number_products = int(params[2].replace('P', ''))
                number_vehicles = int(params[3].replace('V', ''))
                number_periods = int(params[4].replace('T', ''))
                number_seed = int(params[5].replace('S', ''))

                results_file.append({
                    'instance_number':instance_number,
                    'class': get_instance_class(instance_number),
                    'name': name_file,
                    'json': json_data,
                    'seed': number_seed,
                    'numberClients': number_clients,
                    'numberProducts': number_products,
                    'numberVehicles': number_vehicles,
                    'numberPeriods': number_periods
                })

            results.append({
                'dir': directory,
                'resultsFile': results_file
            })

        # Impressão formatada

        for result in results:
            # Cabeçalho da tabela
            header = ["Instância", "Classe", "Seed", "FO", "Gap", "Tempo", "Qtd Solucões", "RLinear", "Branchs", "ObjBound"]
            rows = []

            for file_result in result['resultsFile']:
                print(f"{file_result['name']} - {file_result['class']} - {file_result['seed']}\n")
                rows.append([
                    int(file_result['instance_number']),
                    file_result['class'],
                    int(file_result['seed']),
                    float(file_result['json']['FO']),
                    float(file_result['json']['gap'])*100,
                    float(file_result['json']['time']),
                    int(file_result['json']['solCount']),
                    float(file_result['json']['relaxeModelObjeVal']),
                    int(file_result['json']['nodeCount']),
                    float(file_result['json']['objBound'])
                ])
            
            # Criar DataFrame com cabeçalho correto
            df = pd.DataFrame(rows, columns=header)
            
            # Ordenar e arredondar
            df = df.sort_values(by=['Instância', 'Seed', 'Classe'], ascending=[True, True, True])
            df = df.round(2)


            # Salvar CSV
            df.to_csv(f"out/{result['dir']}/{result['dir']}.csv", index=False)

            # Salvar LaTeX
            latex = df.to_latex(index=False, float_format="%.2f")
            with open(f"out/{result['dir']}/{result['dir']}.txt", "w", encoding="utf-8") as f:
                f.write(latex)

            classes = df['Classe'].unique()
            seeds = df['Seed'].unique()
    
            # Salvar arquivos separados por classe
            for c in classes:
                df_class = df[df['Classe'] == c]
                
                # Caminhos para salvar
                csv_path = f"out/{result['dir']}/{result['dir']}_{c}.csv"
                latex_path = f"out/{result['dir']}/{result['dir']}_{c}.txt"

                # Salvar CSV
                df_class.to_csv(csv_path, index=False)

                # Salvar LaTeX
                latex = df_class.to_latex(index=False, float_format="%.2f")
                with open(latex_path, "w", encoding="utf-8") as f:
                    f.write(latex)
                        # Salvar arquivos separados por classe
            for c in seeds:
                df_class = df[df['Seed'] == c]
                
                # Caminhos para salvar
                csv_path = f"out/{result['dir']}/{result['dir']}_{c}.csv"
                latex_path = f"out/{result['dir']}/{result['dir']}_{c}.txt"

                # Salvar CSV
                df_class.to_csv(csv_path, index=False)

                # Salvar LaTeX
                latex = df_class.to_latex(index=False, float_format="%.2f")
                with open(latex_path, "w", encoding="utf-8") as f:
                    f.write(latex)