
class ReadingFunction:
    def __init__(self,caminho_arquivo):
        self.caminho_arquivo = caminho_arquivo

    def ler_arquivo_oultsp(self):
        dados = {}
        with open(self.caminho_arquivo, 'r') as arquivo:
            linhas = arquivo.readlines()
            # Lendo informações gerais
            comment = ""
            for linha in linhas[1:4]:
                if(len(linha.split(':', 1))==2):
                    comment=comment+" "+linha.split(':', 1)[1].strip()
                else:
                    comment=comment+" "+linha.split(':', 1)[0].strip()
            typeCost = ""
            for linha in linhas[5:7]:
                if(len(linha.split(':', 1))==2):
                    typeCost=typeCost+" "+linha.split(':', 1)[1].strip()
                else:
                    typeCost=typeCost+" "+linha.split(':', 1)[0].strip()
        
            dados['tipo'] = linhas[0].split(':')[1].strip()
            dados['comentario'] = comment
            dados['tipo_custo_transporte'] = typeCost
            dados['nome'] = linhas[7].split(':')[1].strip()
            dados['dimension'] = int(linhas[8].split(':')[1])
            dados['horizon'] = int(linhas[9].split(':')[1])
            dados['capacidade_veiculo'] = float(linhas[10].split(':')[1])
            dados['custo_fixo_transporte'] = float(linhas[11].split(':')[1])

        
            # Lendo informações do fornecedor
            fornecedor_info = linhas[13].split()
            dados['fornecedor'] = {
                'fornecedor': (int(fornecedor_info[0])),
                'coordenadas': (float(fornecedor_info[1]), float(fornecedor_info[2])),
                'nivel_inicial_estoque': int(fornecedor_info[3]),
                'custo_inventario': float(0),
                'custo_variavel_producao': float(fornecedor_info[4]),
                'custo_configuracao': float(fornecedor_info[5]),
                'custo_producao': float(fornecedor_info[6])
            }

            # Lendo informações dos varejistas
            dados['varejistas'] = []
            print(linhas[14])
            for linha in linhas[15:]:
                info_varejista = linha.split()
                varejista = {
                    'varejista': (int(info_varejista[0])),
                    'coordenadas': (float(info_varejista[1]), float(info_varejista[2])),
                    'nivel_max_estoque': float(info_varejista[3]),
                    'nivel_min_estoque': float(info_varejista[4]),
                    'quantidade_estoque': float(info_varejista[5]),
                    'custo_inventario': float(info_varejista[6]),
                    'custo_producao': float(info_varejista[7])
                }
                dados['varejistas'].append(varejista)

            print(dados['varejistas'][0])

        return dados