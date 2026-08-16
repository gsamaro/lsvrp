# Testes

| Arquivo de teste | Tipo | O que valida |
|---|---|---|
| `tests/test_worker_process_mpi.py` | Unitário com executor falso | Uso de um único `MPIPoolExecutor`, submissão de todas as tarefas e fallback sequencial |
| `tests/test_process_results.py` | Unitário com pandas/numpy falsos | Geração de hashes, escrita de `.xlsx`/`.parquet`, ausência de solução e rotas |

Os testes usam mocks para não depender de CPLEX, MPI real ou arquivos Excel reais. Execute a suíte com:

```bash
source "$HOME/.zshrc" && poetry run pytest
```

Em cluster, os scripts PBS executam `mpirun poetry run python -m mpi4py.futures main.py`. O caminho MPI cria um único executor por execução e envia todas as tarefas; diferenças do ambiente MPI real ainda exigem validação no cluster.
