# Matheurística determinística do MPPRP

Implementada sobre a base `0a58f85`. A demanda permanece determinística. O método constrói a solução por etapas e reutiliza o modelo principal, seus objetivos e suas chaves de recursos opcionais.

## Configuração

O bloco é normalizado e validado por `Config`. Chaves ausentes usam estes defaults:

```json
"matheuristic": {
  "enabled": false,
  "use_as": "final",
  "tsp": {"time_limit": 10},
  "restricted": {"time_limit": 60},
  "route_improvement": {"enabled": true, "time_limit": 30},
  "debug": {"enabled": false}
}
```

Esse bloco pertence a `solver`. Os limites são positivos, finitos, expressos em segundos e independentes. O limite existente `solver.timeLimit` continua sendo passado ao solver exato de destino. `solver.pso.time_limit` tem default `null`: quando definido, o PSO verifica a parada entre iterações e conserva sua melhor solução. A inicialização do enxame integra esse tempo; os PLs auxiliares, criados no construtor, e o JIT têm métricas e limites próprios. O exato posterior ao PSO usa seu limite separado.

| `use_as` | `solver.method` | Resultado |
| --- | --- | --- |
| `final` | qualquer método válido existente | melhor solução factível das etapas 2 e 3 |
| `exact_start` | `GUROBY` | solução das etapas como MIP start do exato |
| `pso_start` | `PSO` | solução das etapas como uma partícula completa |

`GUROBY` é o nome histórico do método DOcplex/CPLEX. O fluxo normal é preservado com `enabled=false`. Os controles `solver.pso.use_as_mip_start` e `return_heuristic_result_without_cplex` continuam determinando se o PSO chama o exato posteriormente.

Com a matheurística habilitada, `postprocessing.build_target=true` e `relaxed_solution.use=true` são rejeitados: esses modos substituiriam a formulação das etapas por uma relaxação integral.

## Etapas

1. **TSP:** único, simétrico, sem capacidade e sem índices de produto, veículo ou período. `u[i,k]` é binária. Inicialização por vizinho mais próximo e 2-opt determinísticos; SEC adicionadas por callback lazy. A melhor rota válida encontrada dentro do limite define `pi`; sem incumbente, a rota inicial é utilizada. Cada processo mantém até 64 TSPs em cache LRU, identificado pela matriz de custos, limite e threads.
2. **Restrito:** a ordem define `A^pi = {(0,i)} ∪ {(pi_r,pi_s):r<s} ∪ {(i,0)}`. Somente esses arcos criam `R/Z`; os omitidos são zero. `Z` é contínua entre zero e um, e `eta[v,i,t]` é binária, inclusive no depósito. Entrada e saída são iguais a `eta`, e cada cliente pode ser visitado por no máximo um veículo. Essas igualdades substituem grau, saída máxima e visita antigos. A função configurada é preservada. A construção existente fornece uma tentativa de warm start, reordenada por `pi`, com cargas recalculadas e auditoria de factibilidade.
3. **Melhoria:** fixa `X/Y` e `sum_v Q[p,v,i,t]`, liberando sequência e veículo. `Q/R` continuam contínuas; `Z/eta` são binárias. Só clientes com entrega agregada estritamente positiva podem ser visitados, por exatamente um veículo. A soma dos balanços de produto exclui subtours desses clientes. Períodos sem entregas não criam arcos e têm visitas zero. Minimiza `F4+F5`. A semente remove visitas sem entrega e recalcula as cargas.

A seleção compara **a função configurada**, não o objetivo próprio da etapa 3; em empate numérico de `1e-8`, prefere menor `F4+F5`. Uma melhoria de transporte pode ser descartada quando piora a programação por metas. Sem incumbente restrita, `final` registra ausência de solução e os destinos continuam sem semente. Sem candidato válido na melhoria, conserva a solução restrita. Resultados do destino são auditados antes da aceitação; um resultado ausente ou pior conserva a solução anterior.

### Inicialização e recursos opcionais

O MIP start inclui `X,Y,I,Q,R,Z`, zeros dos arcos omitidos e os desvios/lambda necessários na variante de metas. O nível `WriteLevel.AllVars` preserva quantidades contínuas na transmissão. A semente PSO é inserida diretamente como uma partícula completa, com melhores valores pessoal/global inicializados pela avaliação configurada. As demais partículas mantêm a construção normal. Os intervalos dos genes são ampliados para acomodar a semente dentro dos limites físicos; a semente não precisa ser reconstruída a partir dos genes para ser preservada.

Coelho, HC1, cortes cumulativos de capacidade, bounds fortalecidos e a variante `positive_only_deviations` mantêm as chaves existentes. Quando existe `eta`, as expressões de visita usam essa variável. Callbacks indexam somente arcos criados e interpretam omitidos como zero. HC1 permuta conjuntamente rotas, entregas, cargas e visitas. Os resumos registram recursos aplicados e estatísticas do callback.

### Quantidades e custo computacional

`X,I,Q,R` são contínuas em todos os métodos, inclusive com a matheurística desabilitada. Construção, reparos, estados, materialização e exportação usam `float64`, sem arredondar quantidades. `Y`, atribuições e índices continuam inteiros. `solver.pso.lot_sizing_bounds.integer_variables` é aceita por compatibilidade e normalizada para `false`.

A avaliação numérica da função configurada e dos custos é compartilhada com PSO e resultados. O PSO mantém kernels Numba e estados com rotas compactas; não cria `R/Z` por partícula. Os snapshots entre etapas armazenam valores de arcos em dicionários esparsos, com adjacências pré-calculadas e extração direta das variáveis. Tensores de arcos são materializados para saída, debug ou transmissão que os requer. Cada modelo termina depois da extração. Escrita complementar ocorre em lotes de 4096 registros. TSP e MIPs recebem o limite de threads do worker; o PSO limita suas threads Numba ao limite informado.

## Saídas

A tupla de 16 resultados, índices, hashes, Excel e Parquet finais são preservados. O tempo final da matheurística inclui suas etapas e seu destino. Se o resultado final veio da etapa restrita ou da melhoria, gap/bounds globais são `NaN`: seus gaps não são certificados do problema irrestrito. Quando o PSO conserva sua solução após uma chamada ao exato sem melhoria válida, a telemetria identifica a origem selecionada e não atribui a ela o gap do incumbente descartado.

Cada diretório de saída da instância recebe `matheuristic/`, com um `run_id` exclusivo por execução:

| Arquivo | Conteúdo |
| --- | --- |
| `<run_id>.variables.parquet` | `run_id,hash_file,stage,var,t,p,v,i,k,j,value`; `u` do TSP e `eta` de cada etapa com incumbente; debug acrescenta `X,Y,I,Q,R,Z,P,N,lambda` |
| `<run_id>.stages.parquet` | status, solução disponível/selecionada, objetivos, custos, tempos, limite, tamanho, recursos, bound, gap e `gap_scope` |
| `<run_id>.metadata.json` | configuração efetiva, instância/commit, dimensões, pesos/metas, `pi`, famílias/formatos, debug e parâmetros físicos no debug |

`u/eta` incluem zeros. Famílias físicas e auxiliares de debug são esparsas, com zeros implícitos documentados em `available_variables`; uma família ausente significa que não foi salva. Não há valores para extrair de uma etapa sem incumbente. Uma incumbente rejeitada pela auditoria conserva suas variáveis para inspeção, com `has_incumbent=true`, `has_solution=false` e as violações no resumo. `cache_hit` informa reutilização do TSP; o bound/gap armazenado pertence à resolução original que preencheu o cache.

O arquivo final mantém o nome legado, que pode ser sobrescrito por uma nova execução com os mesmos dados/pesos/alpha. Os metadados complementares registram tamanho e mtime dessa saída: o notebook verifica essa correspondência antes de usá-la. Snapshots de debug permitem inspecionar execuções antigas independentemente da saída final.

## Análise posterior e validação

`analysis/matheuristic_debug.ipynb` lê os arquivos acima, reconstrói TSP, rotas, produção, estoques, entregas e cargas, audita domínios, graus `eta/Z`, balanços, capacidades e conectividade, recalcula custos/função e compara etapas/tempos. Os gráficos permanecem no notebook, que **não é executado automaticamente**.

Execute somente testes unitários, sem resolução CPLEX:

```bash
source "$HOME/.zshrc" && PYTHONPATH=. poetry run pytest --unit-only
```

Esse modo exclui testes marcados `cplex_integration` e bloqueia `Model.solve` real. Os testes do PSO usam um double para os PLs auxiliares. A avaliação real de integração CPLEX/callbacks, velocidade, memória, qualidade e experimentos fica pendente da orientação do usuário.
