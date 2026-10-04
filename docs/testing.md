# Testes

| Arquivo de teste | Tipo | O que valida |
|---|---|---|
| `tests/test_worker_process_mpi.py` | Unitário com executor falso | Uso de um único `MPIPoolExecutor`, submissão de todas as tarefas e fallback sequencial |
| `tests/test_process_results.py` | Unitário com pandas e pyarrow reais | Geração de hashes, escrita de `.xlsx`/`.parquet` e ausência de solução |

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
