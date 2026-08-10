# 3.3 Geração de soluções factíveis

## Estado atual da abordagem

O PSO usa um modelo auxiliar de dimensionamento sem rotas para obter perfis de
produção e entrega. No experimento atual, esse modelo está configurado como PL
contínuo (`integer_variables=false`), com limite de 1 segundo por resolução.
O modo inteiro (`integer_variables=true`) permanece disponível para comparação.
O solver completo de produção e roteamento não participa da avaliação standalone
do PSO.

Após a construção dos déficits obrigatórios, a planta conserva estoque de cada
produto até `U[p][0]`; somente o excedente acima dessa capacidade é enviado aos
clientes. Essa alteração tornou a solução LB do modelo auxiliar factível no PSO
na instância avaliada.

Na medição compilada mais recente, com 1000 partículas e 1000 iterações, todas
as partículas foram factíveis. Houve um perfil de quantidade entregue por
cliente, mas 1000 perfis de atribuição de veículos e tensor completo `Q`. O
melhor custo, `5391676`, foi encontrado na inicialização. O fluxo quente até a
materialização do MIP start levou 31,06 segundos.

## Bounds relaxados de dimensionamento para o PSO

Antes da inicialização do enxame, o PSO pode resolver o PL auxiliar de
dimensionamento implementado em `LotSizingRelaxation`. O modelo mantém apenas
as restrições 8, 9, 10 e 12 do modelo matemático, além da não negatividade,
considerando as variáveis contínuas de produção `X`, estoque `I` e entrega
`Q`. Roteamento, setup e capacidade dos veículos permanecem na etapa de
construção da solução factível.

Para manter os bounds de entrega compatíveis com a frota, o PL também inclui a
aproximação `sum_i sum_p Q[p,v,i,t] <= C` para cada veículo e período. Ela não
modela arcos ou sequência de rota, mas limita a carga total atribuída a cada
veículo.

Agora são resolvidos dois PLs globais sobre as entregas: um minimizando
`sum(Q[p,v,i,t])`, com `i != 0`, e outro maximizando a mesma expressão. O
segundo PL recebe restrições `X[p,t] >= lower_solution['X'][p,t]` e
`Q[p,v,i,t] >= lower_solution['Q'][p,v,i,t]`, garantindo que os perfis upper
não fiquem abaixo dos lower. As matrizes de produção `X` continuam a definir
os bounds de produção; as matrizes `Q` são decodificadas como metas de entrega
na alocação de excedente. Essa preferência é reparada quando necessário para
preservar estoques `U`, carga `C` e a factibilidade das rotas.

Ao construir cada período, a heurística primeiro atribui os clientes com
déficit aos veículos por `best-fit decreasing`, respeitando a visita única e a
carga `C` de cada veículo. Em seguida, distribui somente o excedente que
ultrapassa `U[p][0]`, usando as capacidades residuais desses veículos e o
estoque disponível em cada cliente. A capacidade de estoque da planta é
`U[p][0]`;
o estoque final da planta pode permanecer positivo, desde que não ultrapasse
`U[p][0]`.
As preferências codificadas pelos genes de `Q` ordenam essa alocação
adicional, preservando perfis de entrega diversos entre as partículas. Se o
excedente não puder ser escoado sob essas capacidades, a partícula é inválida.

### Limitação em avaliação

Ainda podem ser geradas partículas infactíveis quando os bounds impõem uma
produção que não pode ser escoada no período. Na inicialização, a reamostragem
é limitada por configuração. Nas iterações seguintes, a heurística preserva a
solução factível anterior da partícula, evitando reamostragem ilimitada e
mantendo o custo do PSO previsível.

O PL usa por padrão o timeout global `solver.timeLimit`. Esse valor pode ser
substituído por `solver.pso.lot_sizing_bounds.time_limit`; o limite é aplicado a
cada resolução de bound e à solução-base opcional.

### Decisão provisória sobre a semente inicial

A primeira partícula é inicializada com a solução `lower_solution`. Portanto,
quando `base_x[p,t] == lower[p,t]`, o gene correspondente fica próximo do
limite inferior mesmo que exista espaço até `upper[p,t]`. Essa decisão é
intencional e será revisitada posteriormente; por enquanto, a diversidade
inicial é fornecida pelas demais partículas aleatórias e pelas atualizações do
PSO.

Também fica pendente avaliar uma formulação alternativa dos bounds, substituindo
os objetivos sobre a produção `sum(X[p,t])` por objetivos sobre as entregas
`Q`. Essa comparação deverá verificar se minimizar/maximizar as entregas produz
bounds mais úteis para a construção das partículas do que os bounds atuais de
produção.

### Melhorias de desempenho e diversidade de `Q`

As partículas são construídas em lote por um kernel Numba. O estado da
população mantém arrays de `X`, `Y`, `I`, `Q`, atribuições, nós das rotas,
comprimentos, custos e factibilidade. Listas, dicionários, `visits`, `R` e `Z`
não são criados no caminho crítico. Uma solução Python individual e os tensores
`R/Z` são materializados somente para um novo melhor global, auditorias, saída
final e warm start.

O backend `numba` é o padrão; `execution_backend=python` preserva uma
implementação de referência para testes. `parallel_workers=auto` usa até oito
threads localmente e uma thread por rank MPI, evitando paralelismo aninhado no
cluster. O aquecimento JIT é medido separadamente e não entra no tempo quente.

O desempate da rota por vizinho mais próximo agora é determinístico: menor
distância, maior preferência agregada `sum_p(raw_q[p,v,i,t])` e menor índice do
cliente. Os backends Python e Numba produzem os mesmos arrays, rotas, custos e
indicadores de factibilidade para as mesmas posições.

O PSO usa inércia e coeficientes cognitivo/social adaptativos, limite de
velocidade e mutação leve em 10% das partículas não elites. Após cinco
iterações sem melhoria, as 15% piores partículas são reinicializadas, mantendo
as 5% melhores. No benchmark de três iterações houve 100 mutações por iteração;
a reinicialização ainda não foi acionada.

Também foi observado que a diversidade medida para `Q` pode ser igual a um:
quando a produção cabe no estoque da planta, as entregas obrigatórias são
determinadas pelos déficits dos clientes e não há entrega adicional para os
genes diferenciarem. Além disso, a métrica atual soma `Q` entre veículos, não
capturando diferenças de atribuição de veículos. Essas duas questões serão
reavaliadas antes de alterar a regra de construção.

A telemetria agora separa `q_quantity_profiles` (quantidades agregadas por
cliente), `vehicle_assignment_profiles` (veículo atribuído a cada cliente e
período) e `q_full_profiles` (tensor completo de `Q`, incluindo veículos).
`q_profiles` permanece como alias compatível de `q_quantity_profiles`.

A telemetria também registra distribuição dos custos, partículas mutadas e
reinicializadas, estagnação, auditorias e tempos de atualização, kernel,
adaptação Python, materialização e cálculo de diversidade. São reportados
separadamente `bounds_seconds`, `jit_warmup_seconds`,
`initialization_seconds`, `pso_hot_elapsed_seconds` e
`pso_cold_elapsed_seconds`.

No benchmark `PRP10_C20_P8_V5_T12_S1`, com oito threads locais, 1000 partículas
e 1000 iterações, foram observados:

- 31,06 s no fluxo quente completo até o MIP start;
- 43,69 s incluindo uma compilação JIT completa de 12,63 s;
- média de 0,0305 s por iteração e percentil 95 de 0,0471 s;
- 100% de factibilidade e 1002 auditorias sem falhas;
- melhor custo `5391676`, encontrado na população inicial.

O benchmark oficial pode ser repetido com
`PYTHONPATH=. poetry run python scripts/benchmark_pso.py`. O MIP exato posterior
não participa do limite de 60 segundos do PSO.

A resolução do problema integrado de produção e roteamento de veículos, uma variante particularmente complexa dos problemas de otimização combinatória, exige a identificação de soluções que atendam simultaneamente a múltiplas restrições operacionais.

Devido à sua natureza NP-difícil e à forte interdependência entre as decisões de produção e roteamento, é fundamental contar com estratégias que possibilitem a geração de soluções iniciais viáveis.

Nesta seção, apresenta-se a abordagem adotada para esse fim, com ênfase em mecanismos que assegurem o atendimento das restrições do problema, como capacidades dos veículos, demandas dos clientes e demais limitações logísticas.

A construção de soluções factíveis representa uma etapa decisiva no desenvolvimento dos métodos de solução, pois permite que os algoritmos atuem desde o início dentro de um espaço de busca admissível.

Detalham-se, a seguir, os procedimentos utilizados para atribuir quantidades de produção ao longo do horizonte de planejamento e para definir rotas de entrega compatíveis com as restrições de capacidade e disponibilidade de produtos em cada período.

Além disso, discutem-se as estratégias de validação de factibilidade adotadas para assegurar que cada solução gerada atenda a todos os requisitos do problema, servindo como ponto de partida robusto para a aplicação dos métodos metaheurísticos propostos nas seções seguintes.

---

# 3.3.1 Heurística construtiva para a geração de soluções

Nesta pesquisa, propõe-se uma heurística que constrói uma solução factível para o problema integrado de produção e roteamento de veículos, de forma iterativa e sequencial ao longo dos períodos de planejamento.

A lógica central baseia-se na interação entre as decisões de produção e estoque e as decisões de roteamento.

Para facilitar a leitura, utilizaremos a mesma notação matemática adotada no modelo apresentado na Seção 3.2.

Uma solução factível e completa para o problema consiste em um conjunto de arranjos, cada um associado a uma variável do problema. A dimensão de cada variável está diretamente relacionada à indexação correspondente.

Por exemplo, as quantidades produzidas \(p_t\) são representadas por um vetor de tamanho \(|T|\), onde cada posição indica a quantidade produzida em cada período do horizonte de planejamento.

*(Figura 5 – Representação da solução)*

A heurística itera sobre cada período de tempo

$$
t \in \{1,\ldots,|T|\}
$$

e executa os seguintes passos.

---

## 1. Decisão de Produção (\(p_t\))

A geração de soluções aleatórias é necessária para garantir a exploração do espaço de busca. Entretanto, essa aleatoriedade pode gerar soluções infactíveis caso não exista controle sobre os intervalos utilizados.

O estoque da fábrica é calculado por

$$
I_t^0 = p_t + I_{t-1}^0 - \sum_{i \in N_c} q_{it}.
$$

Caso o valor de \(p_t\) não seja suficiente para atender às entregas realizadas no período, o estoque poderá tornar-se negativo.

Por esse motivo, é necessário definir um intervalo de geração que preserve a factibilidade.

### Dedução do estoque da fábrica

Sabe-se que

$$
0 \le I_t^0 \le l_0,
\qquad \forall t \in T.
$$

Substituindo a equação do estoque:

$$
0 \le
p_t + I_{t-1}^0 -
\sum_{i=1}^{|N_c|}
q_{it}
\le l_0.
$$

Manipulando a desigualdade:

$$
p_t
\ge
\sum_{i=1}^{|N_c|}
q_{it}
-
I_{t-1}^0.
$$

Portanto, a produção deve satisfazer

$$
p_t
\ge
\sum_{i=1}^{|N_c|}
q_{it}
-
I_{t-1}^0.
$$

Como as entregas \(q_{it}\) ainda são desconhecidas nesse momento, utiliza-se uma estimativa baseada na soma das demandas dos clientes.

Assim,

$$
\underline{p_t}
=
\sum_{i=1}^{|N_c|}
d_{it}
-
I_{t-1}^0,
$$

e a produção é sorteada no intervalo

$$
\underline{p_t}
\le
p_t
\le
M_t.
$$

Para calcular \(\underline{p_t}\), considera-se:

- a demanda dos clientes;
- o estoque inicial da fábrica;
- o estoque disponível dos clientes.

Caso algum cliente possua estoque suficiente para atender sua demanda, essa demanda não é considerada na soma.

---

## 2. Decisão de Entrega (\(q_{it}\)), Estoque (\(I_{it}\)) e Visita (\(s_{it}\))

Para cada cliente \(i\) no período \(t\):

- Se

$$
I_{i,t-1} \ge d_{it},
$$

não é realizada entrega:

$$
q_{it}=0.
$$

O estoque do cliente é dado por

$$
I_{it}
=
q_{it}
+
I_{i,t-1}
-
d_{it}.
$$

Como \(q_{it}\) também é gerado aleatoriamente, é necessário limitar seu intervalo para evitar estoques negativos.

### Dedução do estoque dos clientes

Sabe-se que

$$
0
\le
I_{it}
\le
l_i,
\qquad
\forall i\in N_c,\;
t\in T.
$$

Substituindo a equação do estoque:

$$
0
\le
q_{it}
+
I_{i,t-1}
-
d_{it}
\le
l_i.
$$

Obtém-se

$$
q_{it}
\ge
d_{it}
-
I_{i,t-1}.
$$

Assim, cada entrega deve satisfazer

$$
q_{it}
\ge
d_{it}
-
I_{i,t-1}.
$$

### Intervalo de geração

Para

$$
t<|T|,
$$

gera-se

$$
q_{it}
\sim
\left[
d_{it}-I_{i,t-1},
\;
\min(l_i,\tilde M_{it})
\right].
$$

No último período,

$$
q_{it}
=
d_{it}
-
I_{i,t-1},
$$

garantindo o atendimento completo da demanda remanescente.

### Variável de visita

A variável de visita é definida como

$$
s_{it}
=
\begin{cases}
1, & q_{it}>0,\\
0, & \text{caso contrário.}
\end{cases}
$$

O estoque final do cliente é atualizado por

$$
I_{it}
=
q_{it}
+
I_{i,t-1}
-
d_{it}.
$$

---

## 3. Atualização do estoque da fábrica

Após definir todas as entregas do período, atualiza-se o estoque da fábrica por

$$
I_t
=
p_t
+
I_{t-1}
-
\sum_{i\in N}
q_{it}.
$$

onde:

- \(p_t\) representa a produção do período;
- \(I_{t-1}\) é o estoque do período anterior;
- \(\sum_i q_{it}\) representa o total entregue aos clientes.

---

## 4. Integração com o roteamento de veículos

Após determinar:

- as quantidades entregues (\(q_{it}\));
- os clientes visitados (\(s_{it}\));

essas informações são utilizadas como entrada para uma heurística externa de roteamento baseada no algoritmo do **Vizinho Mais Próximo**.

A heurística utiliza:

- matriz de custos \(c_{ij}^t\);
- capacidade dos veículos \(Q_k\).

Como saída, são construídas as rotas representadas por

$$
x_{ijkt},
$$

que, juntamente com as demais variáveis da solução, compõem uma solução factível para o problema integrado.

O Pseudocódigo 1 apresenta o algoritmo completo de geração das soluções.

---

# 3.3.2 Vantagens e limitações da heurística construtiva

A heurística proposta apresenta como principal vantagem a geração rápida de soluções iniciais factíveis, requisito indispensável para qualquer método de otimização.

A introdução de componentes aleatórios nas decisões de produção e entrega permite gerar uma população inicial diversificada.

Essa diversidade é especialmente importante quando a heurística é utilizada como etapa de inicialização de metaheurísticas populacionais, como o **Particle Swarm Optimization (PSO)**, pois amplia a capacidade de exploração do espaço de busca.

Entretanto, devido à sua natureza construtiva e ao uso de aleatoriedade, as soluções produzidas não são necessariamente de baixo custo.

Seu objetivo principal é garantir **factibilidade** e **diversidade**, servindo como ponto de partida para uma etapa posterior de otimização, e não minimizar diretamente a função objetivo.

## Integração com o módulo PSO

Na implementação do projeto, a heurística construtiva também é utilizada como mecanismo de:

- inicialização de partículas factíveis;
- reparo de partículas que perdem factibilidade após a atualização das posições;
- geração de solução inicial para o `warm start` do CPLEX.

Assim, o PSO opera sobre decisões contínuas de produção e entrega, mas cada partícula é convertida para uma solução completa e factível antes de ser avaliada ou enviada ao solver exato.


## Telemetria persistida e performance profiles

Quando `solver.telemetry.enabled` está ativo, cada tarefa salva shards Parquet em `telemetry/`: um resumo da execução, as métricas por iteração do PSO e, quando aplicável, os eventos MIP de primeira solução factível e alcance do gap-alvo. O script `scripts/consolidate_telemetry.py --input out/` consolida esses dados e cria quatro relatórios Plotly autocontidos: performance profiles de primeira factível, gap-alvo e execução concluída, além da taxa de sucesso.

O cenário `pso_mip_start` preserva separadamente o tempo do PSO, do MIP e o total. Timeouts que não atingem um evento não entram na curva daquele evento, mas permanecem no denominador da taxa de sucesso.
