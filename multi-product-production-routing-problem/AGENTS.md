# INSTRUÇÕES GLOBAIS

## Otimização de Tokens

- Utilize leitura parcial (`offset`/`limit`) em arquivos grandes. Nunca leia um arquivo inteiro quando apenas uma seção for necessária.
- Evite reler arquivos que já foram lidos durante a sessão após uma edição. O ambiente já rastreia o estado dos arquivos modificados.
- Não utilize subagentes para tarefas que possam ser resolvidas com uma ou duas chamadas de ferramenta diretas.
- Prefira buscas locais (`grep`, `find`, `rg`) antes de recorrer a agentes exploratórios para pesquisas simples.
- Mantenha as respostas objetivas. Evite explicações extensas, exceto quando solicitadas explicitamente.
- Evite executar chamadas de ferramentas em paralelo quando houver dependência entre elas. Aguarde o resultado da primeira operação antes de iniciar a próxima.

## Seleção de Modelo

Antes de iniciar qualquer tarefa que não seja trivial, avalie se o modelo atual é adequado.

### Guia de Referência

- **5.4-mini** → buscas rápidas, consultas pontuais, edições simples, consultas SQL/BigQuery básicas e resumos curtos.
- **5.4** → desenvolvimento de código, correção de bugs, tarefas de complexidade média e análises moderadas.
- **5.5** → arquitetura complexa, refatorações envolvendo múltiplos arquivos, análises profundas e design de sistemas.

Se o modelo atual não for o mais indicado para a tarefa, informe ao usuário antes de prosseguir:

> Esta tarefa é mais adequada para o modelo **[modelo recomendado]**. Modelo atual: **[modelo atual]**. Execute `/model [modelo recomendado]` e me avise, ou confirme que deseja continuar com o modelo atual.

Não execute a tarefa até que o usuário confirme a troca ou decida prosseguir com o modelo atual.

## Memória de Sessão e Compactação

Mantenha um contador interno de chamadas de ferramentas realizadas durante a sessão.

A cada aproximadamente 40 chamadas de ferramenta:

1. Gere um resumo da sessão contendo:
   - Objetivo da sessão;
   - Decisões tomadas;
   - Arquivos modificados;
   - Estado atual do trabalho;
   - Próximos passos pendentes.

2. Salve o resumo em:

   `memory/session-current.md`

3. Informe o usuário:

> Sessão longa detectada (~40 chamadas de ferramenta). Um resumo foi salvo em `memory/session-current.md`. Após concluir a tarefa atual, recomenda-se executar `/compact` para reduzir o consumo de contexto.

Nunca interrompa a tarefa em andamento para gerar o resumo. Faça isso apenas ao final da resposta atual.

# CONTEXTO

Para execuções no sandbox que realmente testem o código do projeto, usar `main.py` por meio de `./run_with_zshrc.sh` como ponto de entrada, para garantir que `~/.zshrc` seja carregado antes do Python/Poetry e que o ambiente do CPLEX fique igual ao do bash/zsh local.

Para outras execuções no sandbox, como `git commit`, revisão de arquivos, leitura de status ou tarefas de manutenção que não executem o solver, não é necessário passar por `main.py` nem pelo wrapper `./run_with_zshrc.sh`.

This repository uses `poetry` for Python dependency and environment management.

Esse é um repositório de implementação de um problema de pesquisa operacional usando a biblioteca docplex e resolvido através do solver cplex. Localmene, o problema é resolvido em paralelo e no cluster, em produção, o problema é paralelizado com MPI (não é possível testar a execução MPI localmente).

O Problema de Roteamento de Produção (PRP, do inglês Production Routing Problem), que combina dois problemas clássicos da pesquisa operacional: o Problema de Dimensionamento de Lotes (PDL) e o Problema de Roteamento de Veículos (PRV). O PDL responde à pergunta: quanto e quando produzir cada item, levando em conta custos de preparação de máquina (setup), custos de produção e limites de capacidade fabril. O PRV, por sua vez, determina quais clientes visitar, em qual ordem e com quais veículos, respeitando a capacidade de carga da frota e minimizando os custos de transporte. A integração desses dois problemas em um único modelo de otimização caracteriza o PRP.

---

```
Directory structure:
└── multi-product-production-routing-problem/
    ├── README.md
    ├── constants.py
    ├── main.py
    ├── pyproject.toml
    ├── script_memlong.sh
    ├── script_memshort.sh
    ├── script_paralela.sh
    ├── script_parexp.sh
    ├── script_testes.sh
    ├── .gitingestignore
    ├── .pre-commit-config.yaml
    ├── .python-version
    ├── analysis/
    │   ├── example_vars.ipynb
    │   ├── gap analysis.ipynb
    │   ├── performance profile fob.ipynb
    │   ├── performance profile weights.ipynb
    │   └── sensitivity analysis.ipynb
    ├── config/
    │   ├── __init__.py
    │   ├── config.json
    │   └── config.py
    ├── scripts/
    │   └── generate_reports.py
    └── src/
        ├── helpers/
        │   ├── Converter.py
        │   ├── GraphDisplay.py
        │   ├── InstanceMetadata.py
        │   ├── Outputs.py
        │   ├── ReadPrpFile.py
        │   └── TargetsLoader.py
        ├── log/
        │   └── Logger.py
        ├── process/
        │   ├── InstanceProcess.py
        │   ├── PostProcessingProcess.py
        │   ├── ProcessResults.py
        │   ├── TablesResult.py
        │   └── WorkerProcess.py
        ├── reports/
        │   ├── __init__.py
        │   ├── gap_analysis.py
        │   ├── performance_fob.py
        │   ├── performance_weights.py
        │   ├── sensitivity.py
        │   └── utils.py
        └── solvers/
            ├── GreedyRandomizedConstructionRoute.py
            ├── MultProductProdctionRoutingProblem.py
            ├── MultProductProdctionRoutingProblemGreedyConstructiveHeuristic.py
            └── TwoOptOnRoute.py
```

---
# FILE: README.md

## Exemplos de config.json

Para Executar todas as instancias de uma pasta

    {
        "solver":{
            "threadsLimit":1,
            "timeLimit":10,
            "method":"GUROBY"
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
            "method":"GUROBY"
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

A Classe IV é ignorada no contexto desse repositório.
