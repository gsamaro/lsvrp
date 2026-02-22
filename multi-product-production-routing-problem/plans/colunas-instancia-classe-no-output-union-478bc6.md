# Adicionar colunas de instância e classe no output e no union_results

Este plano adiciona colunas derivadas do nome do arquivo (`file`) aos outputs individuais e ao consolidado (`union_results`), de forma modular e reutilizável.

## 1) Escopo e comportamento desejado
- **Outputs individuais (por execução)**: o Excel gerado em `ProcessResults.getResults` deve conter novas colunas:
  - `instancia`, `clientes`, `produtos`, `veiculos`, `periodos`, `seeds` (extraídas de `file` via regex)
  - `classe` (calculada a partir de `instancia`)
- **Union results (consolidado)**: o Excel gerado por `PostProcessingProcess.union_results` deve conter as mesmas colunas.
- **Compatibilidade**: se o `file` não bater com o padrão, as novas colunas devem existir, porém com `NA`/`None` (sem quebrar o pipeline).

## 2) Design (SOLID) e modularização
- **Single Responsibility**: isolar a lógica de enriquecimento do dataframe (extração + tipagem + classe) em um único ponto.
- **Open/Closed**: permitir evoluir regras de extração (novo pattern) e regras de classificação (novos bins/labels) sem mexer na geração do Excel/union.
- **Dependency Inversion**: `ProcessResults` e `PostProcessingProcess` apenas “chamam” um enriquecedor; eles não devem conhecer detalhes do regex/pd.cut.

### 2.1) Novo módulo utilitário
Criar um helper dedicado (ex.: `src/helpers/InstanceMetadata.py`) com uma API pequena:
- `enrich_with_instance_metadata(df: pd.DataFrame, file_col: str = "file") -> pd.DataFrame`
  - Valida existência de `file_col`
  - Aplica `pattern = r"PRP(\d+)_C(\d+)_P(\d+)_V(\d+)_T(\d+)_S(\d+)"`
  - Cria as colunas `instancia`, `clientes`, `produtos`, `veiculos`, `periodos`, `seeds`
  - Faz cast numérico (com tolerância a erro)
  - Calcula `classe` com `bins=[0,10,20,30,40]`, `labels=[1,2,3,4]`, `include_lowest=True`
  - Retorna **cópia** do df (sem mutar o original), para evitar efeitos colaterais.

## 3) Pontos de integração
### 3.1) Output individual (ProcessResults.getResults)
- Após construir `df_aux` e antes de `to_excel`, chamar `enrich_with_instance_metadata(df_aux)`.
- Garantir que `df_aux['file']` já está preenchida (hoje está como `[data['file']] * n`).

### 3.2) Consolidado (PostProcessingProcess.union_results)
- Após `union_df = pd.concat(...)` e antes de salvar o Excel:
  - Chamar `enrich_with_instance_metadata(union_df)`.
- Isso cobre também casos onde arquivos antigos não tinham as colunas (elas serão recomputadas no consolidado).

### 3.3) Saída alternativa
- Existe também `src/helpers/Outputs.py::_union_results` (redundante em relação ao método de `PostProcessingProcess.union_results`).
  - Estratégia preferida (SOLID / DRY): **centralizar a lógica de union em um único ponto** (`PostProcessingProcess.union_results`).
  - Ação: transformar `_union_results` em um **wrapper fino** (delegando para `PostProcessingProcess.union_results`) ou descontinuá-lo, evitando dois lugares com regras de filtragem/merge.
  - O enrichment (regex + `classe`) permanece em um **helper reutilizável** e é chamado apenas no fluxo central (e, por consequência, por qualquer wrapper).

## 4) Regras de dados e edge cases
- **Quando `file` é path vs basename**:
  - `data['file']` vem como caminho completo com extensão `.dat` (ex.: `./data/.../PRP1_C5_P3_V1_T6_S4.dat`).
  - Extrair o `basename` e remover a extensão antes do regex (ex.: `os.path.splitext(os.path.basename(file))`).
- **Quando o pattern não casa**:
  - Manter colunas e preencher com `pd.NA`; `classe` também `pd.NA`.
- **Tipagem**:
  - `instancia` etc. como `Int64` (nullable) ou `float`/`object` dependendo do padrão adotado no projeto; evitar `.astype(int)` direto se houver `NA`.

## 5) Validação rápida (pós-implementação)
- Rodar um pós-processamento em um output existente e verificar:
  - Excel individual contém as colunas novas.
  - `union_results.xlsx` contém as colunas novas e valores coerentes.
  - Nenhuma exceção ao consolidar arquivos com `file` fora do padrão.

## 6) Checklist de mudanças (arquivos prováveis)
- `src/helpers/InstanceMetadata.py` (novo)
- `src/process/ProcessResults.py` (aplicar enrichment antes do `to_excel`)
- `src/process/PostProcessingProcess.py` (aplicar enrichment antes de salvar union)
- Opcional: `src/helpers/Outputs.py` (se ainda for usado no fluxo)

## 7) Perguntas para confirmar antes de implementar
1) Confirmado: `data['file']` vem como caminho completo com extensão `.dat`; o enrichment deve normalizar para basename sem extensão antes do regex.
2) Confirmado: `instancia` usa o primeiro número após `PRP` (1 a 2 dígitos).
3) Confirmado: não haverá instâncias fora dos `bins`.
