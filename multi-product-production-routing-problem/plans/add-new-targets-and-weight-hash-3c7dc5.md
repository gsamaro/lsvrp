# Adicionar `new_targets` e hash (6-hex) de weights nos Excels

Este plano adiciona as colunas `new_f1_target`..`new_f5_target` (derivadas de `self.new_targets`) ao output/Excels de resultados e cria um hash curto (6 caracteres hex) por conjunto de `weights`, gravando-o como nova coluna para rastrear a rodada.

## Escopo e comportamento esperado

- Os arquivos Excel gerados por instância/rodada (atualmente gerados em `src/process/ProcessResults.py::getResults` via `df_aux.to_excel`) devem conter:
  - Uma nova coluna `weight_hash` com **6 caracteres hex** que identifica o conjunto de `weights` usado naquela execução.
  - Novas colunas por linha/período:
    - `new_f1_target`, `new_f2_target`, `new_f3_target`, `new_f4_target`, `new_f5_target`
- O Excel consolidado (union) gerado em `src/process/PostProcessingProcess.py::union_results` deve **carregar e propagar** essas novas colunas automaticamente (por concatenação), sem quebrar o merge atual de `targets.xlsx`.

## Decisões de implementação

- **Hash 6-hex de weights**
  - Calcular um SHA1 a partir de uma representação **estável** do vetor `weights` (ex.: `"w1,w2,w3,w4,w5"` com normalização de floats), e usar `hexdigest()[:6]`.
  - Nome da coluna: `weight_hash`.
  - Obs.: o arquivo já tem `hash_file` (SHA1 de `file + weight`), então `weight_hash` é especificamente para identificar o conjunto de pesos (independente do arquivo/período).

- **`self.new_targets` no output**
  - `self.new_targets` é criado em `MultProductProdctionRoutingProblem._adjust_targets()` como dict indexado por período `t`.
  - Para o output tabular (1 linha por período), expandir em 5 colunas por linha (com base no `t` da linha):
    - `new_f*_target = self.new_targets[t]["f*_target"]` (ou `NaN` quando indisponível).
  - A fonte de verdade para `new_targets` deve vir do solver (instância), não do pós-processamento.

## Arquivos a alterar

- `src/solvers/MultProductProdctionRoutingProblem.py`
  - Ajustar `getResults()` para também retornar `new_targets` (ou as 5 séries por período), garantindo que:
    - Se `multiobjective` estiver desligado ou `self.new_targets` não existir, retornar `None`/estrutura vazia.

- `src/process/InstanceProcess.py`
  - Atualizar o unpack do retorno de `instance.getResults()` para capturar `NEW_TARGETS`.
  - Passar `NEW_TARGETS` para `ProcessResults.getResults(...)`.

- `src/process/ProcessResults.py`
  - Alterar assinatura de `getResults(...)` para receber `NEW_TARGETS`.
  - Calcular `weight_hash` (6-hex) a partir de `data["weight"]` e adicionar no `df_aux`.
  - Preencher `new_f1_target`..`new_f5_target` por linha usando `NEW_TARGETS` (dict por período) e `time` (índice do período).
  - Garantir compatibilidade:
    - Quando `NEW_TARGETS` for `None`/vazio, preencher colunas com `NaN`.

- `src/process/PostProcessingProcess.py`
  - Nenhuma lógica especial é necessária se as colunas já existirem nos Excels de origem; validar apenas que `union_results` não filtra/descarta as colunas.
  - (Opcional) Garantir que, quando algum arquivo antigo não tiver as colunas, elas sejam criadas com `NA` no `union_df`.

## Testes/validação manual (rápidos)

- Rodar 1 instância com `solver.multiobjective=true` e targets carregados:
  - Verificar se o Excel por instância contém `weight_hash` e `new_f*_target` preenchidos.
- Rodar 1 instância com `solver.multiobjective=false`:
  - Verificar se as colunas existem mas ficam vazias (`NaN`).
- Rodar `PostProcessingProcess.union_results()`:
  - Verificar se o union Excel contém as novas colunas sem conflitos com `f*_target` (targets do arquivo) e sem quebrar o merge.

## Riscos e cuidados

- Alterar a tupla retornada por `MultProductProdctionRoutingProblem.getResults()` exige atualizar todos os call sites (principalmente `InstanceProcess`).
- Garantir que a geração do `weight_hash` seja reprodutível (normalização de floats e ordem fixa do vetor).
