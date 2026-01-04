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
    source path/to/venv/Scripts/activate


bash
Copiar código

    pip install gurobipy
    pip install numpy
    pip install matplotlib
    pip install pandas
    pip install tabulate
    pip install Jinja2

Executar o código dentro do ambiente virtual: Agora você pode executar seu código Python com o gurobipy instalado dentro do ambiente virtual.


## Executar terminal

Para Executar sem travar o terminal atual:

	gnome-terminal -- bash -c "python3 main.py"


## Exemplos de config.json

Para Executar todas as instancias de uma pasta

    {
        "solver":{
            "threadsLimit":1,
            "timeLimit":10,
            "method":"GUROBY/HEURISTICA_CONSTRUTIVA_MIT_START/HEURISTICA_CONSTRUTIVA"
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
            "timeLimit":10,
            "method":"GUROBY/HEURISTICA_CONSTRUTIVA_MIT_START/HEURISTICA_CONSTRUTIVA"
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


## Algoritmo grasp

    função GRASP(Instância, max_iter, α):
        melhor_solucao ← ∅
        melhor_custo ← ∞

        para iter de 1 até max_iter faça:
            # 1. Fase de construção
            solucao ← Construção_Gulosa_Randômica(Instância, α)

            # 2. Fase de busca local
            solucao_melhorada ← Busca_Local(solucao)

            custo ← Avaliar(solucao_melhorada)

            se custo < melhor_custo então:
                melhor_solucao ← solucao_melhorada
                melhor_custo ← custo

        retornar (melhor_solucao, melhor_custo)

    ---------------------------------------------

    função Construção_Gulosa_Randômica(Instância, α):
        solucao ← ∅
        candidatos ← Lista de todos elementos possíveis
        enquanto solução incompleta faça:
            # Avalia custo/benefício de cada candidato
            lista_custo ← Avaliar_Candidatos(candidatos, solucao)

            # Ordena candidatos
            ordenar(lista_custo)

            # Cria RCL (Restricted Candidate List)
            rcl ← primeiros elementos até um limite definido por α

            # Escolhe aleatoriamente um candidato da RCL
            escolhido ← sorteio(rcl)

            adicionar escolhido à solucao
            remover escolhido de candidatos

        retornar solucao

    ---------------------------------------------

    função Busca_Local(solucao):
        melhoria ← verdadeiro
        enquanto melhoria seja verdadeira faça:
            melhoria ← falso
            para cada vizinho em Gerar_Vizinhos(solucao) faça:
                se Avaliar(vizinho) < Avaliar(solucao) então:
                    solucao ← vizinho
                    melhoria ← verdadeiro
        retornar solucao

## A Memetic Algorithm

Algoritmos genéticos, propostos por John Holland em meados da década de 1970, surgiram da analogia entre seleção natural e mecanismos genéticos e de técnicas de busca para obter soluções para problemas de otimização. Em um algoritmo genético, um conjunto de indivíduos, denominado população inicial, é gerado aleatoriamente. A cada indivíduo é atribuído um valor de aptidão, geralmente relacionado ao valor da função objetivo e, portanto, cada indivíduo representa um ponto no espaço de busca do problema. Os operadores de reprodução, cruzamento e mutação são então aplicados a pares de indivíduos (pais) que foram escolhidos por meio de um mecanismo de seleção, dando origem a novos indivíduos (filhos). As repetições do processo produzirão um novo conjunto de indivíduos que formarão uma nova população, inicialmente mais evoluída, considerando que se trata de um modelo evolutivo que incorpora conceitos de sobrevivência e seleção dos mais adaptados.

Ao incorporar estratégias de busca local em algoritmos genéticos, temos
os chamados algoritmos meméticos (Moscato & Norman, 1992) ou
algoritmos genéticos híbridos. Em algoritmos meméticos, os indivíduos passam por
um processo de evolução cultural que ocorre por meio de uma busca local aplicada aos filhos após a execução dos operadores de reprodução.
Nesse caso, os indivíduos são chamados de agentes e possuem informações específicas
do problema que foram incorporadas pela busca local aplicada.

### Algorithm 4: Basic structure of MA

#### Individual

- Production: xpt, ypt, Ipit
- Inventory: Ipit
- Routing: rpvikt, qpvit, zvikt

#### Initial solution:

- **routing problem**

    - BFD (Best-Fit Decreasing)
    - BFD-Inverse (Best-Fit Decreasing Inverse)
    - BFD-Rand (Best-Fit Decreasing Randomized)
    - CW (Clarke & Wright, 1964)
    - CW-Rand

- **production problem**
    - WW: Adapted the Wagner and Whitin (1958) algorithm
    - WW-Rand:To guarantee the generation not only of one but of
several individuals for the initial population,
    - LpL: lot-for-lot heuristic was implemented to obtain
a production plan.

#### Selection and replacement mechanisms
- selectPockets:
- selectCurrents:

#### Repais
- Production capacity violation
- Negative plant inventories:
- Excess production
- Negative customer inventories
- Excess deliveries
- Violation of customer inventory capacity

#### Local search
- backEmptyVehicle
- forEmptyVehicle:
- backwardLoad:
- forwardLoad:
- 2Opt:
- 2OptOut:
- 3Opt:
- swapCustomer:
- relocateCustomer:

###

    Data: Parameters - Instance and MA
    Result: Best individual at population
    Initialize population;
    Update population structure;
    Evaluate diversity;
    while stop == false do
        forall Population do
            if 𝑟𝑎𝑛𝑑.𝑛𝑒𝑥𝑡𝐷𝑜𝑢𝑏𝑙𝑒() < 0.9 then
                A,B ← selectPockets(population);
            else
                A,B ← selectCurrents(population);
            end
            child ← GeneticOperators(A, B);
            if child is not feasible then
                Repair(child);
            end
            if child is feasible then
                LocalSearch(child);
            end
        end
        Update population structure;
        Evaluate diversity;
        if critical diversity achieved then
            Restart population;
            Update population structure;
            Evaluate diversity;
        end
        if stop criterion reached then
            stop ← true;
        end
    end
    return Best individual;






    FAZER UMA SOLUÇÃO INICIAL GRASP
