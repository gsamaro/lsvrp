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


Exemplos de congig.json

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









