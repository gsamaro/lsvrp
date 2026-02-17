# Incluir targets (f1–f5) e alpha no `union_results.xlsx`

Este plano adiciona ao `union_results.xlsx` (o consolidado com `run_tag`, que inclui hash de commit e hash de timestamp) colunas com os targets `f1_target..f5_target` e o valor escalar de `alpha` (`ALPHA[0]`), preenchendo esses campos de forma consistente para cada linha do resultado.

## Contexto (como está hoje)

- `union_results.xlsx` é gerado em `src/process/PostProcessingProcess.py` (método `union_results`) por concatenação de todos os `.xlsx` dentro da pasta `output`.
- Cada execução de instância gera um arquivo `*_fobs.xlsx` via `src/process/ProcessResults.py` (`getResults`). Esse arquivo contém, por período (`time`), as colunas `file`, `time`, `f1..f5`, `weight`, etc.
- Os targets (`f1_target..f5_target`) são calculados em `PostProcessingProcess.build_target()` a partir do próprio `union_results.xlsx` e salvos em `targets.xlsx`.
- Quando `postprocessing.build_target = false`, os targets são carregados de `targets.xlsx` (se existir) pelo `WorkerProcess` e injetados em `InstanceProcess` como `data["targets"]`. O solver (`MultProductProdctionRoutingProblem`) usa esses targets como `self.targets[t]["f*_target"]`.
- O `alpha` do modelo (na parte multiobjetivo) vem de `constants.py` (`ALPHA = [0.01]`) e é atribuído no solver como `self.alpha = ALPHA`.

## Decisão de implementação (o que vai mudar)

- O `union_results.xlsx` passará a conter as colunas:
  - `f1_target`, `f2_target`, `f3_target`, `f4_target`, `f5_target`
  - `alpha`
- Essas colunas serão adicionadas **no pós-processamento do consolidado** (isto é, dentro de `PostProcessingProcess.union_results()`), garantindo que o arquivo consolidado com `run_tag` (commit hash + timestamp hash) já contenha targets e `alpha`.

## Onde alterar

1. `src/process/PostProcessingProcess.py` (método `union_results`)
   - Após `union_df = pd.concat(...)`, adicionar:
     - `alpha`: preencher como valor escalar `ALPHA[0]` (repetido em todas as linhas).
     - `f1_target..f5_target`: realizar `merge` com `targets.xlsx` (se existir) usando as chaves `file` e `time`.
       - `targets.xlsx` é lido de `Config.get_nested("postprocessing", "output")/targets.xlsx`.
       - Se `targets.xlsx` não existir, criar as colunas como vazias (`NaN`).

2. (Opcional) `src/helpers/Outputs.py`
   - Este helper tem uma função `_union_results` similar, mas o fluxo principal usa `PostProcessingProcess`. Não é necessário mudar, a menos que você ainda o utilize em algum lugar.

## Regras de preenchimento dos targets no resultado

Como o `targets.xlsx` é indexado por (`file`, `time`), a lógica será:

- Fazer um `left merge` de `union_df` com `targets_df[["file","time","f1_target",...,"f5_target"]]`.
- Se não houver correspondência para uma linha (`file`,`time`), os `f*_target` ficam `NaN`.

## Questões de compatibilidade / efeitos colaterais esperados

- Execuções com `postprocessing.build_target = true` (modo de construir targets):
  - O `targets.xlsx` pode não existir antes do `build_target()`, então os `f*_target` no consolidado podem ficar vazios. Isso é OK, pois o `build_target()` calcula targets a partir de `f1..f5` do próprio `union_results`.
- Execuções normais (otimização):
  - Se `targets.xlsx` existir e bater com (`file`,`time`), os targets serão preenchidos no consolidado com `run_tag`.
- O valor de `alpha` será sempre incluído como escalar (`ALPHA[0]`).

## Validação rápida após implementação

- Rodar uma execução pequena e abrir o `out/...-union_results.xlsx`.
- Verificar:
  - Presença das colunas `f1_target..f5_target` e `alpha`.
  - Para instâncias com `targets.xlsx` disponível, conferir se os targets batem com `out/target/targets.xlsx` para o mesmo (`file`, `time`).
