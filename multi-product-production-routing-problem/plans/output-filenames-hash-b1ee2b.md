# Output filenames com data + hashes + fluxo de targets
Este plano cria um identificador único de execução para prefixar o `union_results.xlsx`, ajusta o fluxo de targets no `main.py` e garante que o `*_fobs.xlsx` contenha o hash correto (`file_name_hash`).

## Contexto (onde mexer)
- `src/process/PostProcessingProcess.py`
  - `union_results()` grava hoje `union_results.xlsx`.
  - `build_target()` lê fixo `union_results.xlsx` (vai precisar ser ajustado para ler o novo nome).
- `src/process/ProcessResults.py`
  - `getResults()` grava `df_aux` em `f"{file_name_hash[:6]}_fobs.xlsx"`.
- `main.py`
  - Orquestra execução e chama `postprocessing.union_results()` e opcionalmente `build_target()`.

## Mudança requerida
1. Nome do relatório consolidado:
   - De: `union_results.xlsx`
   - Para: `yyyy-mm-dd-{hash_commit6}-{hash_timestamp6}-union_results.xlsx`
2. Fluxo de targets no `main.py` deve ser separado em dois:
   - Se `config.json/postprocessing/build_target=true`: rodar `build_target()` e salvar **`targets.xlsx`** (sem hash no nome).
   - Se `config.json/postprocessing/build_target=false`: não construir targets (o uso automático de `target.xlsx` ainda não está implementado; ignorar neste escopo).
3. `*_fobs.xlsx`:
   - Não adicionar nenhuma coluna nova (a coluna `hash_file` já existe e deve permanecer).

## Estratégia proposta (identificador único da execução)
- Calcular no início do `main.py` um **run tag** único.
- **Não usar variável de ambiente**.
- Armazenar o `RUN_TAG` como atributo em cada item do dicionário dentro da lista `instancies` (ex.: chave `run_tag`) para ficar disponível durante o processamento das instâncias.
- Componentes do `RUN_TAG`:
  - `date`: `datetime.now().strftime('%Y-%m-%d')`
  - `hash_commit6`: 6 primeiros caracteres do `git rev-parse HEAD` (fallback seguro se não for repo git / comando falhar)
  - `hash_timestamp6`: 6 primeiros do `sha1(timestamp_completo)` (timestamp completo em ISO, ex.: `2026-02-07T10:40:00.123456-03:00`)
- String final (exemplo): `2026-02-07-acde12-9f3a01`

## Alterações por arquivo
### 1) `main.py`
- Gerar `RUN_TAG` uma vez no início.
- Ao construir a lista `instancies`, adicionar `"run_tag": RUN_TAG` em cada dicionário.
- Ajustar o pós-processamento em 2 fluxos:
  - Se `build_target=true`:
    - chamar `postprocessing.union_results(run_tag=RUN_TAG)` (ou equivalente) e depois `postprocessing.build_target(...)`.
    - salvar o arquivo de saída como `targets.xlsx`.
  - Se `build_target=false`:
    - não chamar `build_target()`.

### 2) `src/process/PostProcessingProcess.py`
- `union_results()`:
  - Usar `RUN_TAG` para compor o nome do arquivo de saída.
  - Gravar em `os.path.join(self.output, f"{RUN_TAG}-union_results.xlsx")`.
  - Retornar o `out_path` como já retorna hoje.
- `build_target()`:
  - Parar de assumir `union_results.xlsx`.
  - Receber `out_path` como parâmetro (ou usar o último caminho retornado por `union_results()` armazenado na instância) e ler esse caminho.
  - Salvar o arquivo final de targets como `targets.xlsx` (sem hash no nome).

### 3) `src/process/ProcessResults.py`
- `getResults()`:
  - Nenhuma alteração no dataframe do `*_fobs.xlsx` (manter `hash_file` como está hoje).

### 4) (Opcional) `src/helpers/Outputs.py`
- Se este helper ainda for usado em algum fluxo, manter consistente com a mesma regra de nome (ou remover o uso em favor do `PostProcessingProcess`).

## Perguntas (para confirmar antes de implementar)
1. O `union_results.xlsx` com prefixo deve ser salvo diretamente dentro de `./out/` (como hoje) ou dentro de algum subdiretório específico?

## Validação após implementação
- Rodar uma execução pequena.
- Checar se foi criado exatamente um arquivo:
  - `output/<RUN_TAG>-union_results.xlsx`
- Checar qualquer `*_fobs.xlsx` gerado:
  - continua contendo a coluna `hash_file`.
- Se `build_target` estiver habilitado:
  - confirmar que ele lê o novo `union_results` e gera `targets.xlsx`.
