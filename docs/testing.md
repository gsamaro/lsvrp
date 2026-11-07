# Testes

| Arquivo de teste | Tipo | O que valida |
|---|---|---|
| `tests/test_worker_process_mpi.py` | Unitário com executor falso | Uso de um único `MPIPoolExecutor`, submissão de todas as tarefas e fallback sequencial |
| `tests/test_process_results.py` | Unitário com pandas e pyarrow reais | Geração de hashes, escrita de `.xlsx`/`.parquet` e ausência de solução |
| `tests/test_pso_decoder.py` | Unitário | Opções do decodificador do PSO: validação, equivalência entre o kernel Numba e o backend Python (limiar de setup, economias, 2-opt, entregas antecipadas, bounds e frotas apertadas), períodos sem setup, agrupamento por vizinhança, retorno ao best-fit, nivelamento de carga que torna viável uma instância inviável "na hora", genes de cobertura e seus cortes (estoque, `C`, frota, produção), viés inicial e configuração |
| `tests/test_matheuristic_fleet.py` | Unitário e integração CPLEX | Modos de partida da matheurística, limpeza dos totais de entrega, `arc_order` e plano fixo sem `eta`, tamanho menor dos modelos sem veículo, igualdade do ótimo do estágio restrito com e sem índice (mono e multiobjetivo), melhoria que preserva o plano e fluxo só matheurística |
| `tests/test_experiment_runner.py` | Unitário | Vários experimentos num job: ordem, pasta e estado por experimento, retomada pelo `union_results` do experimento correto (nome, fingerprint, nº de tarefas; sem glob), pasta antiga movida, falha isolada, validação prévia, chaves de pareamento `task_key`/`row_key` entre experimentos |
| `tests/test_time_budget_polishing.py` | Unitário | Orçamento total por tarefa (`TimeBudget`, `timeLimit` limitado ao restante) e `solver.polishing.after_seconds` (só no MIP completo, relógio do CPLEX, mínimo de 1 s) |
| `tests/test_objective_scale.py` | Unitário e integração CPLEX (`cplex_integration`) | Fator `solver.objective_scale`: padrão neutro, coeficientes escalados no CPLEX, ramos sem escala (objetivo único), `_unscale` e igualdade de `FO`, limite, gap e desvios entre os modelos com e sem escala |
| `tests/test_symmetric_routes.py` | Unitário e integração CPLEX (`cplex_integration`) | `solver.symmetric_routes` e `goal_programming.optimize_f1`: validação de config, escopo (etapas restritas, matriz simétrica), forma do modelo, caminhada das arestas, igualdade de ótimo com o modelo direcionado (objetivo único e multiobjetivo, várias capacidades), MIP start convertido (válido e inválido), saída em `Z`/`R`/`Q` auditável, relaxação de targets, pesos efetivos, produção mínima, `trim_surplus` e entrega da matheurística ao MIP simétrico |
| `tests/test_validate_targets.py` | Unitário | `validate_targets`: arquivo ausente, instância sem alvos ou sem todos os períodos, execuções que não usam alvos e parada dos experimentos antes de rodar |
| `tests/test_first_need_setup_cuts.py` | Unitário e integração CPLEX (`cplex_integration`) | `first_need_periods` (estoque de clientes e da planta), uma desigualdade por produto, flag e `fixed_plan`, validação da config e do bloco de geração de targets, ótimo do MIP inalterado e LP de setup maior que o anterior |
| `tests/test_vehicle_aggregation.py` | Unitário e integração CPLEX (`cplex_integration`) | Formulação sem índice de veículo: configuração, precedência da opção, forma do modelo, divisão de rotas por veículo, MIP start e igualdade do ótimo com o modelo indexado |
| `tests/test_disconnected_cycles.py` | Unitário e integração CPLEX (`cplex_integration`) | Remoção de ciclos desconectados da planta na solução extraída do MIP (`drop_disconnected_arcs`): arcos e carga removidos, rotas da planta e períodos/slots vizinhos intactos, valores não binários deixados à auditoria, solução limpa aprovada pela auditoria e sem custo maior, leitura de um incumbente falso em `get_snapshot` e `getResults` (objetivo, desvios e `lambda` recalculados); o modelo admite o ciclo sem entrega (agregado e indexado, mono e multiobjetivo, com e sem carga circulante) e rejeita o ciclo com entrega |
| `tests/test_rejected_solutions.py` | Unitário | Auditoria `validate_variables` acumulando todas as violações (falha de rota não esconde as demais), tolerância absoluta de capacidade, e rastro da incumbente do MIP descartada no PSO (log, registro `exact` em `get_artifacts`, telemetria) |

Os testes usam mocks para não depender de CPLEX ou MPI real. Os testes de persistência usam arquivos Excel e Parquet reais em diretórios temporários. Execute a suíte com:

```bash
source "$HOME/.zshrc" && PYTHONPATH=. poetry run pytest --unit-only
```

Em cluster, os scripts PBS executam `mpirun poetry run python -m mpi4py.futures main.py`. O caminho MPI cria um único executor por execução e envia todas as tarefas; diferenças do ambiente MPI real ainda exigem validação no cluster.


`--unit-only` exclui os testes marcados `cplex_integration` e bloqueia chamadas
reais a `docplex.mp.model.Model.solve`. Nos testes de comportamento do PSO,
os PLs auxiliares usam um double. A montagem de modelos DOcplex, a avaliação
numérica Python/Numba e a escrita de arquivos temporários são testadas sem
resolver otimizações. `tests/test_matheuristic.py` cobre configuração, destinos,
quantidades fracionárias, TSP/cache/SEC, domínios/graus, planos fixos, callbacks
esparsos, HC1, seleção multiobjetivo, falta de incumbente, seeds, prazo PSO e debug.

Testes CPLEX reais, instâncias, benchmarks, experimentos e o notebook de
diagnóstico não foram executados na implementação da matheurística.

Validação em 2026-10-03: **200 testes passaram e 4 testes de integração CPLEX
foram excluídos**, usando `--unit-only`. O LaTeX existente compilou com sucesso
no editor. Os avisos restantes da suíte são de depreciação de pandas nos relatórios.
