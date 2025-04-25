## Criação do Ambiente

Criar um ambiente virtual: No terminal, execute o seguinte comando para criar um ambiente virtual em um diretório específico (substitua path/to/venv pelo caminho onde você deseja criar o ambiente):

bash
Copiar código

    python3 -m venv path/to/venv

Ativar o ambiente virtual:

No Linux ou macOS:
bash
Copiar código

    source path/to/venv/bin/activate


bash
Copiar código

    pip install gurobipy
    
Executar o código dentro do ambiente virtual: Agora você pode executar seu código Python com o gurobipy instalado dentro do ambiente virtual.


## Executar terminal

Para Executar sem travar o terminal atual:
	
	gnome-terminal -- bash -c "python3 main.py"


## Exemplos de congig.json

Para Executar todas as instancias de uma pasta

    {
        "solver":{
            "threadsLimit":1,
            "timeLimit":10
        },
        "workers":{
            "num":4,
            "timeSupervisor":1
        },
        "instance":{
            "is_plot": "false",
            "dir": "./data/",
            "output": "./out/",
            "files": ["DATA_PRP_5C"]
        }
    }

Para uma instancia expecifica 

    {
        "solver":{
            "threadsLimit":1,
            "timeLimit":10
        },
        "workers":{
            "num":4,
            "timeSupervisor":1
        },
        "instance":{
            "is_plot": "false",
            "dir": "./data/",
            "output": "./out/",
            "files": ["DATA_PRP_5C/PRP1_C5_P2_V1_T2_S1.dat","DATA_PRP_5C/PRP1_C5_P3_V1_T6_S1.dat"]
        }
    }


## Instancias

Descrições dos Tipos de Classes

| Classes    |  Tipo   |   Descrição                                      |
| ---------- | ------- | -------------------------------------------------|
| Classe I   |  1–10   |   Instâncias padrão                              |
| Classe II  |  11–20  |   Custos de produção elevados (Classe I × 10)    |
| Classe III |  21–30  |   Custos de transporte elevados (Classe I × 5)   |
| Classe IV  |  31–40  |   Sem custos de estoque no cliente               |

Além disso, o primeiro grupo de instâncias possui apenas 1 veículo, o segundo grupo tem 2 veículos e os dois seguintes possuem 5 veículos. As demandas são variáveis e os estoques iniciais dos clientes não são zero. A capacidade de produção da planta é limitada, e a capacidade de armazenamento é ilimitada, mas os estoques iniciais são zero.

Assim como em Archetti et al. (2011), dividimos os grupos em quatro classes de acordo com a Tabela 4. A Classe I (instâncias de 1 a 10) possui a configuração básica de custos de produção, estoque e transporte, servindo como base para a geração das demais. A Classe II (11 a 20) possui altos custos de produção, equivalentes aos custos da Classe I multiplicados por 10. A Classe III (21 a 30) apresenta altos custos de transporte, ou seja, os custos serão 5 vezes maiores do que na Classe I. Por fim, a Classe IV (31 a 40) não possui custos de estoque no cliente. Cada classe possui 10 instâncias com 5 sementes cada; portanto, temos 200 instâncias para cada grupo, totalizando 800 novas instâncias no conjunto como um todo.