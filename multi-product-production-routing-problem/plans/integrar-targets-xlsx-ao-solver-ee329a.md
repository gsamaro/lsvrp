# Integração do `targets.xlsx` ao solver (MPPRP)

Este plano adiciona a leitura de um `targets.xlsx` (uma única vez, fora do solver e fora do `main.py`) e injeta os valores de target dentro da classe `MultProductProdctionRoutingProblem` através do mesmo dicionário `data`/`map` que já é passado pelo pipeline.

## Contexto (como o solver é instanciado hoje)

- `main.py`
  - Monta a lista `instancies` (cada item contém `file`, `output`, `numThreads`, `timeLimit`, etc.).
  - Executa `WorkerProcess(...).run_parallel(instancies=instancies, solver=method)`.
- `WorkerProcess.process(...)` -> cria `InstanceProcess(...).process()`.
- `InstanceProcess.process()`
  - Lê a instância `.dat` via `ReadPrpFile(...).getDataSet()`.
  - Injeta `data["weight"] = self.weight`.
  - Instancia o solver (`MPPRP(map=data, ...)`).
- `MultProductProdctionRoutingProblem.__init__` recebe `map` e consome os campos (ex.: `d_pit`, `a_ik`, `weight`, etc.).

## Decisão principal (melhor forma de trafegar os dados)

- **Estratégia recomendada: carregar `targets.xlsx` uma vez no `WorkerProcess` e injetar no `data` em `InstanceProcess`**.
  - Motivo:
    - IO fica fora do solver.
    - Não polui o `main.py` com funções/loader.
    - Permite cache (carrega uma vez por execução) e reuso para todas as instâncias.
    - Aproveita o padrão já existente de “adicionar campos” ao `data` (como já ocorre com `weight`).

## Contrato de dados proposto

- O arquivo `targets.xlsx` contém uma coluna `file` que identifica a instância.
- Ao carregar, transformar em um dicionário de lookup:
  - `targets_by_file[file] = [linha1_dict, linha2_dict, ...]`
- No momento de processar a instância `.dat`:
  - Definir `data["targets"] = targets_by_file.get(data["file"], [])`
  - *Observação*: `ReadPrpFile` já seta `data["file"] = self.file_path` (caminho do `.dat`). Isso é a chave mais consistente para lookup.

## Onde ler o `targets.xlsx`

- Fonte do path (única, sem mudar `config.json`):
  - Usar `Config.get_nested("postprocessing", "output")` (ex.: `./out/target/`) e compor `targets.xlsx` dentro desse diretório.
  - Ex.: `targets_path = os.path.join(postprocessing_output, "targets.xlsx")`.

## Mudanças planejadas (alto nível)

1) **Criar um pequeno loader de targets (módulo utilitário em `src/helpers/`)**
- Implementar módulo (ex.: `src/helpers/TargetsLoader.py`) com uma função pública (ex.: `load_targets_by_file(targets_path, log)`), que:
  - Recebe `targets_path`.
  - Usa `pandas.read_excel(..., engine="openpyxl")`.
  - Normaliza a coluna `file` (string) e retorna `dict` no formato `targets_by_file`.
- Requisito adicional (para não perder linhas):
  - Se houver **múltiplas linhas** para o mesmo `file`, o loader deve **acumular** (append) ao invés de sobrescrever.
- Comportamento esperado:
  - Se o arquivo não existir: retornar `{}` e logar `warning` (targets são opcionais).
  - Se existir e faltar a coluna `file`: lançar exceção clara (contrato inválido).

2) **Carregar uma única vez no `WorkerProcess` (cache por execução)**
- Em `WorkerProcess.run_parallel(...)` (ou no `__init__`), carregar `targets_by_file` uma vez:
  - `post_out = Config.get_nested("postprocessing", "output")`
  - `targets_path = os.path.join(post_out, "targets.xlsx")`
  - `targets_by_file = load_targets_by_file(targets_path, log)`
- Como o worker chama `InstanceProcess(...)`, passar `targets_by_file` como parâmetro do `InstanceProcess`.

3) **Injetar no dataset da instância**
- Em `InstanceProcess.process()`:
  - Após ler o `.dat` em `data`, anexar `data["targets"]` com base em `data["file"]`.
  - `data["targets"]` deve ser uma **lista** de linhas (dicionários), não um único dicionário.

4) **Consumir os targets dentro do solver**
- Em `MultProductProdctionRoutingProblem.__init__`:
  - Ler `map.get("targets")` e armazenar em `self.targets`.
  - Definir defaults caso não exista target para o `file`.

## Alternativas consideradas

- **Ler o Excel dentro de `MultProductProdctionRoutingProblem.__init__`**
  - Não recomendado: acopla IO ao solver, dificulta paralelismo/testes e repete leitura.
- **Usar `Config` como singleton dentro do solver**
  - Menos explícito e mais difícil de testar; além disso, você ainda precisa resolver o cache/lookup.

## Pontos abertos / validações necessárias

- O valor em `targets.xlsx:file` (confirmado) é um caminho relativo como:
  - `./data/DATA_PRP_5C/PRP1_C5_P3_V1_T6_S1.dat`
- Já o `data["file"]` vindo do `ReadPrpFile` é o `file_path` usado na leitura; dependendo de como `InstanceProcess` é chamado, ele pode ser:
  - relativo (ex.: `./data/...`) ou
  - absoluto (ex.: `C:\\...\\data\\...`).

Portanto, o loader deve **normalizar chaves** para garantir match, por exemplo:
- Normalizar separadores (`os.path.normpath`).
- Reduzir para um formato canônico comum, como:
  - `relpath` a partir do diretório de dados (`Config.get_nested("instance", "dir")`, hoje `./data/`), ou
  - fallback para `basename` se necessário.

## Critérios de aceite

- Para uma instância cujo `file` exista no `targets.xlsx`, `MultProductProdctionRoutingProblem` recebe `self.targets` com as colunas `*_target`.
- Se o `targets.xlsx` tiver múltiplas linhas para o mesmo `file`, todas as linhas devem chegar no solver (via `self.targets` como lista).
- Para instância sem targets, o solver continua rodando sem erro (targets opcionais).
- Leitura do `targets.xlsx` acontece **uma única vez** por execução (não por instância).
