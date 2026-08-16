# Backlog de melhorias pós-simplificação MPI

Este backlog reúne oportunidades identificadas após a remoção de batches,
guardrails e código legado do pipeline MPI. Os itens preservam o contrato atual
de resultados Excel/Parquet, MPI opcional, CPLEX/DOcplex, PSO e telemetria.

## Prioridade 1 — confiabilidade e redução de custo

### BL-01 — Aplicar `timeLimit` ao ciclo PSO

**Problema.** `ParticleSwarmOptimization.solver()` recebe `timeLimit`, mas o
ciclo de iterações do PSO não o consulta. O limite atualmente alcança apenas o
solver exato executado após o PSO, quando aplicável.

**Mudança proposta.** Criar um deadline no início do pipeline e interromper o
PSO entre iterações quando ele expirar. A definição de produto necessária é se
`solver.timeLimit` limita todo o pipeline (PSO + CPLEX) ou somente cada etapa;
a implementação deve refletir essa decisão explicitamente.

**Critérios de aceite.**

- Um teste com relógio controlado prova que o PSO encerra antes de executar a
  próxima iteração após o deadline.
- A telemetria registra corretamente duração, status e quantidade de iterações
  executadas.
- Quando houver CPLEX após o PSO, o tempo entregue a ele segue a semântica
  definida para o limite global ou por etapa.

**Arquivos principais.** `src/solvers/ParticleSwarmOptimization.py`,
`src/helpers/SolverTelemetry.py`, `tests/test_pso_solver.py`.

### BL-02 — Remover o retorno detalhado legado de `ProcessResults`

**Status.** Endereçado no PR [#12](https://github.com/gsamaro/lsvrp/pull/12).

**Problema.** O pipeline grava os arquivos e descarta o retorno de
`getResults()`, porém a função ainda reconstrói rotas e uma estrutura detalhada
de períodos em memória. Testes unitários são os únicos consumidores internos
do retorno identificado.

**Mudança proposta.** Manter a escrita Excel/Parquet e retirar a construção do
payload legado, incluindo reconstrução de rotas, ou torná-lo explicitamente
opt-in se houver consumidor externo confirmado.

**Critérios de aceite.**

- O fluxo `InstanceProcess` não armazena nem apaga um retorno de escrita que
  não utiliza.
- As colunas, hashes e formatos de Excel e Parquet permanecem inalterados.
- Testes passam a verificar os arquivos produzidos, não o payload legado.
- A busca global confirma que nenhum consumidor ativo depende de `periods` ou
  da reconstrução de rotas retornada.

**Arquivos principais.** `src/process/ProcessResults.py`,
`src/process/InstanceProcess.py`, `tests/test_process_results.py`.

### BL-03 — Não gerar targets quando a consolidação falhar

**Problema.** `union_results()` retorna `None` quando não encontra ou não lê
arquivos Excel. Com `build_target=true`, a etapa seguinte tenta usar esse valor
como caminho e produz uma falha secundária, menos clara.

**Mudança proposta.** Em `main.py`, somente chamar `build_target()` quando a
consolidação retornar um caminho válido; registrar o erro operacional e encerrar
com status apropriado.

**Critérios de aceite.**

- Sem planilhas de entrada, não há tentativa de leitura de caminho `None`.
- A mensagem identifica que a consolidação falhou antes da geração de targets.
- Com consolidação válida, o comportamento atual de criação de `targets.xlsx`
  é preservado.

**Arquivos principais.** `main.py`, `src/process/PostProcessingProcess.py`,
novo ou ampliado teste unitário de entrada principal/pós-processamento.

### BL-04 — Corrigir a documentação arquitetural obsoleta

**Problema.** `AGENTS.md` ainda descreve guardrails, batches, `TablesResult`,
`GraphDisplay` e opções de plot como componentes ativos, embora tenham sido
removidos.

**Mudança proposta.** Atualizar arquitetura, fluxos, checklists, glossário e
listas de arquivos/classes para o executor MPI único e o fluxo sequencial de
fallback. Revisar também referências remanescentes em README e documentos de
teste.

**Critérios de aceite.**

- `rg` não encontra referências ativas a guardrails, batch, `timeSupervisor`,
  `TablesResult`, `Outputs`, `is_plot` ou `isPloat` na documentação de
  manutenção.
- Diagramas e tabelas descrevem `MPIPoolExecutor` único e o fallback sem
  `mpi4py`.
- As instruções de teste correspondem ao comando efetivamente usado pelo
  projeto.

**Arquivos principais.** `AGENTS.md`, `README.md`, `docs/testing.md`.

## Prioridade 2 — simplificação estrutural

### BL-05 — Resolver pesos no momento da execução

**Problema.** `WEIGHTS` é escolhido durante a importação de `WorkerProcess`, a
partir de `postprocessing.build_target`. Isto acopla o comportamento a cache e
ordem de importação da configuração.

**Mudança proposta.** Substituir a constante de módulo por um helper que decide
entre `WEIGHTS_TARGET` e `WEIGHTS_OPTIMIZE` dentro de `run_parallel()`.

**Critérios de aceite.**

- A escolha de pesos é feita uma vez por execução de `run_parallel()`.
- Cobertura testa os dois valores de `build_target` sem recarregar o módulo.
- Produto cartesiano de instâncias × pesos × alpha permanece idêntico.

**Arquivos principais.** `src/process/WorkerProcess.py`,
`tests/test_worker_process_mpi.py`.

### BL-06 — Simplificar a interface de `WorkerProcess`

**Problema.** O construtor é anotado para `Logger`, mas exige o formato
`{"instancia": log}` e extrai a chave internamente.

**Mudança proposta.** Receber um `Logger` diretamente e ajustar a entrada
principal e os testes. Se a compatibilidade externa for necessária, planejar
uma transição explícita em vez de manter o contrato implícito.

**Critérios de aceite.**

- `WorkerProcess(numWorkers, log, run_tag=...)` é a única interface interna.
- Não há acesso a `log["instancia"]`.
- MPI e fallback sequencial mantêm os logs e a telemetria atuais.

**Arquivos principais.** `main.py`, `src/process/WorkerProcess.py`,
`tests/test_worker_process_mpi.py`.

### BL-07 — Extrair descoberta de instâncias de `main.py`

**Problema.** A entrada principal concentra leitura de configuração, descoberta
de arquivos, filtro de instâncias PRP e criação de diretórios.

**Mudança proposta.** Extrair uma função pequena e testável, por exemplo
`build_instances(...)`, que recebe diretório, saída e seleção de arquivos. Dar
nome e documentação ao filtro atual de PRPs 1–30 ou torná-lo configuração
explícita, após confirmar se ele é uma regra experimental vigente.

**Critérios de aceite.**

- Casos de diretório e arquivo `.dat` individual têm testes unitários.
- Arquivos não elegíveis não geram diretórios de saída.
- A execução principal fica limitada a orquestração de alto nível.

**Arquivos principais.** `main.py`, `tests/test_main_import.py` ou novo teste.

### BL-08 — Limpar redundâncias e comentários mortos do modelo

**Problema.** O modelo contém longos blocos de logs/prints comentados, um loop
que monta strings de `Z` sem utilizar o resultado e atribuição duplicada de
`self.log`.

**Mudança proposta.** Remover somente comentários de diagnóstico inativos e
código sem efeito, sem renomear a API do modelo nem alterar restrições,
variáveis ou objetivo.

**Critérios de aceite.**

- Busca estática não encontra o loop de strings descartadas ou os blocos
  comentados de impressão de solução.
- Testes de objetivo e resultados do modelo continuam passando.
- A formulação matemática e os nomes públicos existentes não mudam.

**Arquivos principais.**
`src/solvers/MultProductProdctionRoutingProblem.py`,
`tests/test_mpprp_objective.py`.

## Prioridade 3 — robustez de configuração e testes

### BL-09 — Centralizar defaults e validar configuração

**Problema.** Há consultas espalhadas a `Config.get_nested()` e chaves
ausentes podem assumir comportamentos implícitos — por exemplo,
`relaxed_solution.use` é lida pelo modelo, mas não consta no JSON atual.

**Mudança proposta.** Definir defaults documentados e validação leve na camada
`Config`, sem introduzir dependência de validação nova. Mapear primeiro todas as
chaves realmente utilizadas e decidir o comportamento esperado de cada uma.

**Critérios de aceite.**

- Cada chave lida pelo pipeline possui valor documentado ou default central.
- Configuração inválida gera erro claro antes da execução do solver.
- Testes cobrem valor ausente e valor inválido dos controles relevantes.

**Arquivos principais.** `config/config.py`, `config/config.json`, `main.py`,
solvers e testes de configuração novos.

### BL-10 — Enxugar os mocks legados do teste MPI

**Problema.** `test_worker_process_mpi.py` injeta módulos que `WorkerProcess`
não importa diretamente, incluindo um módulo de gráfico já removido.

**Mudança proposta.** Manter somente os doubles de executor, futures e função
`process` necessários para cobrir submissão MPI, workers resolvidos e fallback
sequencial.

**Critérios de aceite.**

- O teste não altera `sys.modules` para módulos não usados pelo módulo sob
  teste.
- Continua cobrindo um executor, todas as submissões e o solver no fallback.

**Arquivos principais.** `tests/test_worker_process_mpi.py`.

## Ordem recomendada

1. BL-01, BL-03 e BL-04.
2. BL-02, com medição de memória/tempo em uma instância representativa antes e
   depois.
3. BL-05 a BL-08.
4. BL-09 e BL-10.

Antes de cada remoção, executar busca global de consumidores; após cada lote,
validar o JSON e rodar `source "$HOME/.zshrc" && PYTHONPATH=. poetry run pytest`.
