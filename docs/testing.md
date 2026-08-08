
## Estratégia de testes

### Testes existentes

| Arquivo de teste | Tipo | O que valida | Evidência principal |
|---|---|---|---|
| `tests/test_job_guardrails.py` | Unitário | Thresholds de walltime/memória, descoberta de `mpi_size`/`mpi_rank`, fallback de `/proc/self/status`, formatação de snapshot, refresh de identidade local | `src/helpers/JobGuardrails.py` |
| `tests/test_instance_process_watchdog.py` | Unitário com doubles | Emissão de logs do watchdog, logs de identidade/checkpoints e cleanup do watchdog quando o solver falha | `src/process/InstanceProcess.py` |
| `tests/test_process_results.py` | Unitário com pandas/numpy falsos | Geração de hashes, escrita de `.xlsx`/`.parquet`, comportamento sem solução, serialização mínima de colunas | `src/process/ProcessResults.py` |
| `tests/test_worker_process_mpi_batching.py` | Unitário com executor falso | Cálculo de tamanho de lote, cap de instâncias pesadas, uso de `MPIPoolExecutor`, divisão em múltiplos batches | `src/process/WorkerProcess.py` |

### Cobertura

| Área | Status observado | Observação |
|---|---|---|
| Guardrails | Cobertura boa para regras locais e metadados de runtime | Foco em thresholds e composição de logs |
| Batching MPI | Cobertura boa para planejamento de lotes | Usa mocks, não executa MPI real |
| `InstanceProcess` | Cobertura parcial | Cobre watchdog e cleanup, não o fluxo completo de solver/extração real |
| `ProcessResults` | Cobertura parcial | Cobre cenários sem solução e schema mínimo de saída |
| Solver principal DOcplex/CPLEX | Cobertura ausente no pacote | Não há teste exercitando restrições, objetivo ou solve real |
| Parser `.dat` | Cobertura ausente no pacote | Não há teste dedicado para `ReadPrpFile` |
| Pós-processamento | Cobertura ausente no pacote | Não há teste para `union_results()` ou `build_target()` |
| Relatórios | Cobertura ausente no pacote | Não há teste para `src/reports/*.py` |
| Heurísticas | Cobertura ausente no pacote | Não há teste para GRASP/2-opt/warm start |

Não foi possível determinar a partir do código uma métrica numérica de cobertura, como percentual de linhas ou branches, porque não há relatório de cobertura incluído no material empacotado.

### Testes unitários

Os testes presentes são majoritariamente unitários e usam isolamento agressivo:

- `sys.modules` é sobrescrito para trocar módulos pesados por dublês em `test_instance_process_watchdog.py` e `test_worker_process_mpi_batching.py`.
- `numpy`, `pandas` e `orjson` são simulados em `test_process_results.py`.
- `unittest.mock.patch` é usado para controlar tempo, ambiente MPI, executor e retorno de funções internas.

Esse desenho sugere que a estratégia atual prioriza validar lógica de orquestração e contratos de saída sem depender de CPLEX, MPI real ou arquivos Excel reais.

### Integração

| Nível de integração | Situação atual | Evidência |
|---|---|---|
| Integração entre `WorkerProcess` e `InstanceProcess` | Parcialmente simulada | `tests/test_worker_process_mpi_batching.py` substitui `process()` |
| Integração entre `InstanceProcess` e solver | Simulada | `tests/test_instance_process_watchdog.py` usa `FakeSolverInstance` |
| Integração entre solver e `ProcessResults` | Não coberta com solver real | testes substituem solver ou estruturas numéricas |
| Integração entre `main.py` e pipeline completo | Não coberta | não há teste envolvendo `main.py` |
| Integração entre pós-processamento e resultados reais | Não coberta | não há teste para `PostProcessingProcess` |
| Integração entre relatórios e Excel consolidado real | Não coberta | não há teste para `scripts/generate_reports.py` ou `src/reports/*.py` |
| Integração MPI real | Não coberta | `AGENTS.md` do repositório afirma que MPI local não é testável |

Na prática, o repositório atual tem uma base de testes unitários focada em lógica crítica de coordenação, mas não uma suíte de integração end-to-end dentro do material fornecido.

### Componentes críticos

| Componente crítico | Por que é crítico | Estado de teste atual |
|---|---|---|
| `src/solvers/MultProductProdctionRoutingProblem.py` | Define objetivo, restrições e solve | Sem teste direto no pacote |
| `src/process/WorkerProcess.py` | Controla explosão combinatória de tarefas e batching MPI | Parcialmente coberto |
| `src/process/InstanceProcess.py` | Coordena leitura, solver, extração e cleanup | Parcialmente coberto |
| `src/process/ProcessResults.py` | Define contrato consumido por pós-processamento e relatórios | Parcialmente coberto |
| `src/helpers/ReadPrpFile.py` | Define contrato de entrada do problema | Sem teste direto |
| `src/process/PostProcessingProcess.py` | Gera `union_results.xlsx` e `targets.xlsx` | Sem teste direto |
| `src/helpers/TargetsLoader.py` | Faz matching entre instância e targets | Sem teste direto |
| `src/reports/utils.py` e `src/reports/*.py` | Dependem do schema final e de colunas derivadas | Sem teste direto |

### Testes que deveriam existir

| Prioridade | Teste recomendado | Motivo |
|---:|---|---|
| 1 | Teste unitário de `ReadPrpFile.read()` com fixture `.dat` mínima | Garantir contrato de parsing do formato de instância |
| 2 | Teste de `TargetsLoader.load_targets_by_file()` com normalização de caminhos | Evitar targets silenciosamente não aplicados |
| 3 | Teste de `PostProcessingProcess.union_results()` com múltiplos `.xlsx` | Garantir merge, colunas padrão e enriquecimento de metadata |
| 4 | Teste de `PostProcessingProcess.build_target()` | Validar geração de targets ideal/nadir |
| 5 | Teste do filtro de instâncias em `main.py` | Garantir exclusão intencional de `PRP >= 31` |
| 6 | Teste unitário de `MultProductProdctionRoutingProblem.crateObjectiveFunction()` | Validar ramos `build_target`, `multiobjective` e single-objective |
| 7 | Teste estrutural do solver com modelo pequeno | Verificar criação de variáveis/restrições sem depender de dataset grande |
| 8 | Teste para `src/reports/utils.py::read_results_df()` | Validar exclusão de pesos unitários, labels e colunas derivadas |
| 9 | Teste de pelo menos um gerador de relatório | Validar pré-condições e arquivos de saída |
| 10 | Smoke test do pipeline via `main.py` com mocks | Cobrir orquestração de ponta a ponta |

### Como executar

| Cenário | Comando | Observações |
|---|---|---|
| Suite unitária padrão | `python -m unittest discover tests` | Compatível com a estrutura presente em `tests/` |
| Arquivo isolado | `python -m unittest tests.test_job_guardrails` | Útil para focar em uma área |
| Execução real do projeto | `./run_with_zshrc.sh` | Recomendado pelo `AGENTS.md` original para carregar ambiente do CPLEX |
| Execução HPC/MPI | `mpirun python -m mpi4py.futures main.py` | Aparece nos scripts PBS; não implica testabilidade local |

Não foi possível determinar a partir do código se existe comando oficial com Poetry para testes, nem se a suíte foi integrada a CI.

### Riscos atuais

| Risco | Impacto | Evidência |
|---|---|---|
| Solver principal sem teste direto | Regressões matemáticas podem passar despercebidas | ausência de testes para `src/solvers/MultProductProdctionRoutingProblem.py` |
| Parser `.dat` sem teste | Mudanças no formato de entrada podem quebrar execução cedo | ausência de testes para `src/helpers/ReadPrpFile.py` |
| Pós-processamento sem teste | `union_results.xlsx` e `targets.xlsx` podem quebrar relatórios em cascata | ausência de testes para `src/process/PostProcessingProcess.py` |
| Relatórios sem teste | Mudanças de schema podem falhar só em fase tardia | ausência de testes em `src/reports/*.py` |
| Testes usam mocks extensivos | Alguns bugs de integração real podem não ser capturados | `tests/test_*.py` |
| MPI real não é exercitado | Diferenças de ambiente/cluster podem só aparecer em produção | `AGENTS.md`, `script_*.sh`, `tests/test_worker_process_mpi_batching.py` |
| Dependência operacional de CPLEX | Falhas de ambiente podem ser confundidas com falhas de código | `run_with_zshrc.sh`, `pyproject.toml`, `AGENTS.md` |
| Contrato de saída é central e frágil | Pequenas mudanças em colunas afetam pós-processamento e relatórios | `src/process/ProcessResults.py`, `src/process/PostProcessingProcess.py`, `src/reports/utils.py` |
