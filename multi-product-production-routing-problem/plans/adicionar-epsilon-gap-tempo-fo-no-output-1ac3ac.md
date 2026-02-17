# Adicionar epsilon, gap, tempo e FO no output

Este plano adiciona ao output tabular (`*_fobs.xlsx` e, por consequência, `union_results.xlsx`) os campos `epsilon`, `gap`, `solver_time` e `objective_value` (FO) provenientes do solver.

## Contexto (como está hoje)

- O solver `MultProductProdctionRoutingProblem` já calcula:
  - `self.time` (tempo de solve)
  - `self.model.solve_details.mip_relative_gap` (gap)
  - `self.model.objective_value` (FO / valor da função objetivo)
- Esses valores já são retornados por `MultProductProdctionRoutingProblem.getResults()` como `FO`, `GAP`, `TIME`.
- O pipeline de escrita de output principal por instância é `src/process/ProcessResults.getResults()`:
  - Escreve um Excel por instância/peso `*_fobs.xlsx` com colunas por período (`time`, `file`, `hash_file`, `f1..f5`, `weight`, `hash_row`, etc.).
- O `union_results.xlsx` é um `concat` dos `*_fobs.xlsx` no `PostProcessingProcess.union_results()`.

## Definição do epsilon

- No modelo multiobjetivo, o “epsilon” a reportar será o **valor solucionado da variável `lambda_`** (nome no modelo: `lambda`).
- Extração prevista:
  - Quando houver solução: `epsilon = float(self.lambda_.solution_value)` (com fallback seguro para `None/0` se não existir solução).

## Mudança requerida (o que será adicionado)

Adicionar as colunas abaixo ao `*_fobs.xlsx` (todas repetidas em todas as linhas/períodos do arquivo da instância):

- `epsilon`: valor do `lambda_` (ou vazio se não multiobjetivo / sem solução)
- `gap`: `self.model.solve_details.mip_relative_gap`
- `solver_time`: `self.time`
- `FO`: `self.model.objective_value`

## Onde alterar

1. `src/solvers/MultProductProdctionRoutingProblem.py`
   - Em `getResults()`:
     - Calcular `epsilon` (ver regra acima).
     - Incluir `epsilon` no tuple retornado (idealmente próximo de `FO/GAP/TIME`).
     - No branch `solCount==0`, retornar `epsilon` como `0` ou `None` para manter consistência.

2. `src/process/InstanceProcess.py`
   - Ajustar o unpack do retorno de `instance.getResults()` para capturar o novo campo `EPSILON`.
   - Passar `EPSILON` para `ProcessResults.getResults()`.

3. `src/process/ProcessResults.py`
   - Atualizar assinatura de `getResults(..., FO, GAP, TIME, ..., EPSILON, ...)` (ou inserir o parâmetro em posição estável) e:
     - Adicionar no `df_aux` as colunas `FO`, `gap`, `solver_time`, `epsilon` (valores constantes por linha/período).

## Compatibilidade / impactos

- O `union_results.xlsx` automaticamente herdará as novas colunas, pois é um `concat` dos `*_fobs.xlsx`.
- Em execuções não-multiobjetivo, `epsilon` ficará vazio (ou `0`/`NaN`), mas as demais colunas (`gap`, `solver_time`, `FO`) continuam válidas.

## Validação rápida após implementação

- Rodar uma instância pequena.
- Abrir um `*_fobs.xlsx` e verificar:
  - Colunas `FO`, `gap`, `solver_time`, `epsilon` presentes.
  - `solver_time` bate aproximadamente com logs/tempo observado.
  - `gap` bate com o `mip_relative_gap` esperado.
- Abrir `out/<RUN_TAG>-union_results.xlsx` e confirmar as mesmas colunas presentes.
