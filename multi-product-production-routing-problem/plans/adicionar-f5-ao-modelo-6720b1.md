# Adicionar f5 ao modelo (impactos no projeto)

Este plano mapeia os pontos do código que precisam ser ajustados para suportar 5 funções objetivo (`f1..f5`) alinhadas ao modelo e refletidas nos relatórios `*_fobs.xlsx` e `union_results.xlsx`.

## Escopo funcional (definições finais)
- **Agora existem 5 funções objetivo**:
  - `f1`: custo de produção
  - `f2`: custo de setup
  - `f3`: custo de estoque
  - `f4`: custo de ativar caminhão (custo fixo)
  - `f5`: custo da rota (custo variável)
- **Relatórios**:
  - `*_fobs.xlsx` e `union_results.xlsx` devem conter **todas as colunas `f1..f5`** alinhadas com essas definições.
- **Execução em bateria**:
  - Deve existir peso para otimizar também `f5` (vetor unitário `[0,0,0,0,1]`), além dos demais.
- **Implementação**:
  - Manter a captura do pipeline via `instance.getResults()` (não trocar por `new_get_results()` no fluxo principal).
  - Estender `getResults()` para computar e salvar `f5`.
  - Ajustar a função `new_get_results()` (em `ProcessResults.py`) para comportar `f5` (mesmo que ela não seja o caminho principal).

## 1) Modelo/solver (função objetivo)
- **Arquivo**: `src/solvers/MultProductProdctionRoutingProblem.py`
- **Local**: método `crateObjectiveFunction()`
- **O que conferir/ajustar**:
  - Garantir que `self.f5` exista e esteja incluída em `objExpr`.
  - Garantir que `self.weight` tenha **5 posições** e que o acesso `self.weight[4]` não estoure.
  - Garantir que `new_get_results()` do solver continue retornando `f5`.

## 2) Vetor de pesos (dimensão e experimentos)
- **Arquivo**: `constants.py`
- **Local**: `WEIGHTS`
- **O que mudar**:
  - Atualizar de vetores 4D para 5D, por exemplo:
    - `[1,0,0,0,0]`, `[0,1,0,0,0]`, `[0,0,1,0,0]`, `[0,0,0,1,0]`, `[0,0,0,0,1]`.
  - Manter pelo menos os unitários (incluindo `f5`) para reproduzir a bateria atual.

- **Arquivo**: `src/process/WorkerProcess.py`
- **Local**: `run_parallel()`
- **O que verificar**:
  - Ele itera `for w in WEIGHTS` e injeta `weight=w` na `InstanceProcess`. Ao aumentar `WEIGHTS` para 5D, ele automaticamente passa a testar `f5` também.

## 3) Cálculo e persistência de resultados (Excel e JSON)
O pipeline de resultados deve ser ajustado para ficar **1:1 com o modelo** (5 componentes separadas).

- **Arquivo**: `src/process/ProcessResults.py`
- **Local**: `getResults(...)`
- **Situação atual**:
  - Gera colunas `f1..f4` no `*_fobs.xlsx`, porém com significado antigo:
    - `f1 = csetup + cprod`
    - `f2 = estoque`
    - `f3 = ativar caminhão (fixo)`
    - `f4 = rota (variável)`
- **O que mudar** (alinhamento novo `f1..f5`):
  - Ajustar o cálculo e a exportação no `df_aux` para produzir as 5 colunas conforme o modelo:
    - `f1 = cprod`
    - `f2 = csetup`
    - `f3 = estoque`
    - `f4 = ativar caminhão (fixo)`
    - `f5 = rota (variável)`
  - Garantir que `*_fobs.xlsx` carregue essas colunas e que `union_results.xlsx` as preserve.

- **Arquivo**: `src/process/InstanceProcess.py`
- **Local**: `process()`
- **Decisão já tomada**:
  - Manter o fluxo atual baseado em `instance.getResults()` + `ProcessResults.getResults()`.
  - Não trocar para `instance.new_get_results()` no pipeline principal.

- **Arquivo**: `src/process/ProcessResults.py`
- **Local**: `new_get_results(...)`
- **O que mudar**:
  - Incluir o parâmetro `f5` e a coluna `f5` no `result.xlsx`.
  - Manter consistência de nome/ordem com `f1..f5` do novo padrão.

## 4) Pós-processamento (target/anti-ideal)
- **Arquivo**: `src/process/PostProcessingProcess.py`
- **Local**: `build_target()`
- **O que mudar**:
  - Atualizar os `pivot_table(... aggfunc=...)` para incluir `f5`.
  - Criar também `f5_target` da mesma forma que `f1_target..f4_target`.

## 5) União de resultados
- **Arquivo**: `src/process/PostProcessingProcess.py` (e também `src/helpers/Outputs.py`, se estiver em uso)
- **Local**: `union_results()` / `_union_results()`
- **Nota**:
  - A união em si concatena DataFrames; ela já tolera colunas novas.
  - Mas qualquer etapa que **assume** a lista de colunas (`f1..f4`) precisa ser atualizada (o `build_target` é o principal).

## Checklist de alterações (arquivos)
- **`constants.py`**: tornar `WEIGHTS` 5D e incluir o peso unitário de `f5`.
- **`src/solvers/MultProductProdctionRoutingProblem.py`**: garantir objetivo com 5 componentes e acesso seguro a `weight[0..4]`.
- **`src/process/ProcessResults.py`**:
  - `getResults(...)`: recalcular e exportar `f1..f5` alinhadas ao modelo (e incluir `f5` no `df_aux`).
  - `new_get_results(...)`: incluir `f5` no `result.xlsx`.
- **`src/process/PostProcessingProcess.py`**: incluir `f5` no `build_target()`.
