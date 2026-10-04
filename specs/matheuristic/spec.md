# Matheurística determinística do MPPRP

Plano aprovado: implementação sobre 0a58f85, com testes unitários apenas.

## Contrato

- `solver.matheuristic.enabled=false` preserva a seleção normal do solver.
- `use_as` aceita `final`, `exact_start` e `pso_start`; os destinos de inicialização exigem `solver.method` correspondente.
- Limites independentes: TSP 10 s, restrito 60 s e melhoria 30 s; melhoria habilitada e debug desabilitado por padrão.
- TSP simétrico único, inicializado por vizinho mais próximo/2-opt, com SEC durante a busca e aceitação de rota válida sem prova de ótimo.
- Restrito: ordem total, Z contínua, eta binária, graus iguais a eta, quantidades contínuas e função configurada.
- Melhoria: X/Y e entregas agregadas fixos, Z/eta binárias, visitas com entrega positiva e objetivo F4+F5.
- Selecionar menor função configurada; empate favorece menor transporte. Sem incumbente restrita, destinos continuam sem semente e modo final registra ausência de solução.
- MIP start completo; PSO recebe uma partícula completa e conserva a semente como incumbente. `solver.pso.time_limit=null` preserva parada por iterações; quando definido, parar entre iterações.
- X/I/R/Q são float64 em todos os solvers, inclusive com a matheurística desabilitada; Y/Z/eta/u são decisões binárias conforme a etapa. A chave legada de quantidades inteiras da relaxação é normalizada para false.
- Coelho, cortes, HC1, bounds e variante de metas conservam suas chaves. Adaptar visitas e índices de arcos esparsos. Matheurística não admite build_target nem relaxed_solution.use.
- Interface de resultados final inalterada. Parquets complementares por run_id: u/eta sempre; variáveis físicas intermediárias e auxiliares apenas no debug, com zeros implícitos documentados.
- Registrar custos, objetivos, status, tamanho, tempos, bound/gap por etapa e cache. Gaps restritos não representam o problema irrestrito.
- Notebook posterior para rotas, produção, estoques, fluxos, capacidades, conectividade e custos; sem análises gráficas no fluxo normal.
- Cache TSP LRU de 64 entradas por processo; modelos liberados entre etapas; índices pré-calculados, extração direta e kernels Numba preservados.

## Validação autorizada

Testes unitários com mocks e verificações numéricas. Não resolver modelos CPLEX, executar instâncias reais, notebooks, benchmarks ou experimentos. Cobrir configuração, três destinos, quantidades fracionárias, Python/Numba, graus/domínios, cortes esparsos, HC1, seleção multiobjetivo, ausência de incumbente, seed PSO, deadline, saídas e debug.

## Evidência de implementação — 2026-10-03

- Implementação na branch `codex/deterministic-matheuristic`, a partir de `0a58f85`; alterações não commitadas.
- `pytest --unit-only -q`: **200 passed, 4 deselected**. Os quatro testes excluídos resolvem modelos CPLEX. O modo unitário bloqueia `Model.solve` real; simulações de solver usam mocks explícitos.
- A execução carregou `~/.zshrc` e reutilizou o ambiente Poetry existente. Os oito avisos da suíte são de depreciação de pandas nos relatórios existentes.
- LaTeX existente editado em seu caminho original e compilado com sucesso pelo compilador do editor.
- Notebook entregue sem células executadas. Instâncias reais, integração real dos callbacks CPLEX e avaliações de tempo, memória e qualidade aguardam orientação do usuário.
