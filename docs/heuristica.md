# 3.3 Geração de soluções factíveis

## Estado atual da abordagem

O PSO usa um modelo auxiliar de dimensionamento sem rotas para obter perfis de
produção e entrega. No experimento atual, esse modelo está configurado como PL
contínuo (`integer_variables=false`), com limite de 1 segundo por resolução.
A chave legada `integer_variables=true` é aceita e normalizada para `false`;
as quantidades de produção, estoque, entrega e carga são contínuas em todos os métodos.
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

### Opções do decodificador: setup e roteamento

O decodificador transforma os genes em produção, entregas e rotas. Uma análise
na instância `PRP13_C20_P8_V5_T12_S4` (multiobjetivo, `alpha = 0,99`, pesos 0,2)
encontrou três limitações, tratadas pelas opções `solver.pso.decoder`.

**Diagnóstico.**

- A produção extra é `sigmoid(gene) * folga`, sempre positiva. Mesmo com gene
  `-60` a quantidade vale `~1e-27`, e `Y = 1` quando a produção é positiva. A
  partícula não tinha como deixar de produzir um produto num período: o melhor
  PSO fazia 96 setups (todo produto, todo período) ou 60, conforme os bounds.
- Os genes de `Q` (10.080 dos 10.176 genes) não alteravam o custo. As entregas
  são `déficit = max(0, demanda - estoque)`, apenas o excedente acima de
  `U[p][0]` é drenado, e a preferência dos genes só desempata. Sortear todos os
  genes de `Q` 200 vezes deu sempre o mesmo custo, e a diversidade de perfis de
  entrega da população era 1.
- Com os bounds do PL ligados, o custo do melhor PSO (`1,2013`) era exatamente
  o custo com todos os genes de produção em `-60`: o PSO devolvia o vértice do
  PL (`X = lower`), pois `producao = max(necessário, lower)` fixa um piso de
  produção, e não melhorava nada além disso. O enxame convergia em ~25
  iterações.
- A atribuição por `best-fit decreasing` ignora a geografia e as rotas eram
  construídas por vizinho mais próximo, sem melhoria.

**Opções** (`solver.pso.decoder`; os defaults de `FeasibleParticleHeuristic`
reproduzem o decodificador antigo, os do PSO e do `Config` ligam as melhorias):

| Opção | Valores | Efeito |
|---|---|---|
| `setup_threshold` | `[0, 1)`, padrão `0.5` | Se `sigmoid(gene)` é no máximo o limiar, a partícula não produz além do necessário no período (a parte acima do limiar é reescalada para `[0, 1]`). Extras abaixo de `1e-6` viram zero. `0` reproduz o decodificador antigo. |
| `delivery` | `just_in_time`, `lookahead` (padrão no PSO) | `just_in_time` entrega só o déficit do período (decodificador antigo). `lookahead` também entrega antes do déficit, ver abaixo. |
| `delivery_threshold` | `[0, 1)`, padrão `0.5` | Limiar do gene de cobertura: abaixo dele o cliente só recebe o déficit. |
| `max_cover` | inteiro `>= 1`, padrão `2` | Máximo de períodos à frente que a cobertura de um cliente pode atender. |
| `fleet_utilization` | `(0, 1]`, padrão `0.9` | Fração da capacidade da frota (`V·C`) que o nivelamento e a cobertura aceitam usar por período. |
| `routing` | `nearest_neighbor`, `savings_2opt` (padrão) | `savings_2opt` particiona os clientes com carga em rotas pelo algoritmo de economias de Clarke-Wright, com capacidade `C` e no máximo `V` rotas, e aplica 2-opt e realocação de um nó (Or-opt) a cada rota. Se a partição usar mais de `V` rotas, o período volta ao `best-fit`. Exige distâncias simétricas; caso contrário usa `nearest_neighbor` com aviso. |

O backend Python é o espelho exato do kernel Numba: mesmas economias, mesma
ordem de desempate (`mergesort` estável sobre pares em ordem crescente de
índice), mesmo 2-opt e mesma subtração das capacidades residuais em ordem
crescente de cliente. `tests/test_pso_decoder.py` compara os dois backends
nos oito modos (limiar, roteamento e bounds), com bounds de entrega e em
frotas apertadas.

**Resultado** (PSO de 1.000 partículas e 1.000 iterações, custo da função
configurada):

| Configuração | Custo na S4 | Setups |
|---|---|---|
| Antes (bounds do PL ligados) | 1,2013 | 60 |
| Sem bounds do PL | 0,5562 | 96 |
| Sem bounds + limiar de setup | 0,4786 | 48 |
| Sem bounds + economias e 2-opt | 0,5397 | 96 |
| Sem bounds + as duas melhorias | **0,3454** | 50 |
| Com bounds + as duas melhorias | 1,1751 | 44 |

Em 12 instâncias de 20 clientes (3 por classe e sementes variadas, 300
iterações) o custo médio caiu de `0,728` (antes) para `0,276` (razão média
`0,40`, entre `0,24` e `0,57`), com solução viável em todas.

**MIP completo com o PSO como solução inicial** (S4, `alpha = 0,99`, bounds
fortalecidos, 5 minutos de MIP, 2 threads, quatro execuções simultâneas):

| PSO | MIP | Solução inicial | Objetivo final | Bound | Gap |
|---|---|---|---|---|---|
| antigo | com índice de veículo | 1,2013 | 1,2013 | 0 | 100% |
| antigo | `aggregate_vehicles` | 1,2013 | 0,5284 | 0,1586 | 70% |
| novo | com índice de veículo | 0,3454 | 0,3454 | 0 | 100% |
| novo | `aggregate_vehicles` | 0,3454 | **0,2405** | **0,1685** | **30%** |

Com o índice de veículo a relaxação da raiz consumiu ~290 s dos 300 s, e o MIP
nem produziu bound. Com `aggregate_vehicles` ela leva ~2 s, o CPLEX aplica
rodadas de cortes e melhora a solução inicial, mas continua na raiz (0 nós) nos
300 s. É uma instância e uma semente; o PSO levou 124-144 s por rodar
simultaneamente com as outras execuções (a compilação do kernel conta
~20 s).

**Bounds do PL.** Eles são o principal peso morto: com as duas melhorias ligadas
o custo médio é `0,677` com bounds e `0,276` sem. Também não garantem
viabilidade: das 801 instâncias do repositório, 103 não têm nenhuma partícula
viável (2 de 10C, 45 de 30C, 56 de 5C, nenhuma de 20C), e os bounds do PL não
resgatam nenhuma delas. `config/config.json` passa a usar
`lot_sizing_bounds.enabled = false`; o default do código continua `true`.

### Entregas antecipadas (`delivery: lookahead`)

No modo `just_in_time` o cliente só recebe o déficit do período. Isso deixava
103 das 801 instâncias do repositório sem nenhuma partícula viável (2 de 10C,
45 de 30C, 56 de 5C): em 68 delas a carga de algum período passa da capacidade
da frota `V·C` (até 1,51×), e em 35 a carga fica em 95-100% da frota e os
clientes não cabem nos veículos (cada cliente é atendido por um só veículo). A
produção nunca era o problema (no máximo 0,44 de `B`).

O modo `lookahead` acrescenta entregas além do déficit, de duas fontes, e
todas são cortadas em vez de invalidar a partícula:

1. **Nivelamento de carga.** Só é ligado nas instâncias em que a carga "na
   hora" de algum período passa de `fleet_utilization·V·C`. A cada período
   calcula a necessidade futura acumulada de cada cliente e produto (acima do
   estoque projetado), faz uma passada de trás para frente com a capacidade
   `fleet_utilization·V·C` por período e obtém quanto precisa sair agora. Esse
   volume é alocado pelos prazos mais próximos primeiro e repartido entre os
   clientes proporcionalmente à necessidade, o que mantém as cargas parecidas
   e fáceis de empacotar. Fica limitado pela folga da frota no período e pelo
   estoque máximo de cada cliente.
2. **Genes de cobertura.** O gene de `Q` do par `(cliente, período)` (slot
   `p=0, v=0`; 252 dos 10.080 genes na S4) acima de `delivery_threshold` faz o
   cliente receber o necessário para cobrir os próximos `k <= max_cover`
   períodos, o que permite pular visitas e setups. A quantidade é limitada por
   `U` do cliente, por `C` e pela capacidade usável da frota.

As entregas são feitas antes da produção, que passa a produzir sob demanda; se
a antecipação excede o tempo de produção `B`, só a parte antecipada é
reduzida. O kernel Numba e o backend Python seguem o mesmo algoritmo
(`tests/test_pso_decoder.py`; os floats diferem no máximo por arredondamento,
pois a multiplicação-soma da alocação proporcional pode ser fundida no
kernel).

**Viés inicial** (`solver.pso.initial_bias`). Os genes começam em `N(0, 1)`,
então metade dos genes de entrega começa ligada e a população parte longe do
plano "na hora" (na S4, 0,465 contra 0,345). Os genes de entrega são
deslocados em `-delivery` (padrão 3) e os de produção em `-production` (padrão
1), só quando o limiar correspondente está ativo, e isso vale para a
inicialização, a reamostragem e a reinicialização por estagnação.

**Resultado.**

- **Viabilidade** (300 partículas aleatórias por instância, 801 instâncias):

  | Grupo | Sem partícula viável: `just_in_time` → `lookahead` | Fração viável média |
  |---|---|---|
  | 5C | 56 → 0 | 0,721 → 1,000 |
  | 10C | 2 → 0 | 0,990 → 1,000 |
  | 20C | 0 → 0 | 1,000 → 1,000 |
  | 30C | 45 → 0 | 0,775 → 1,000 |

  A fração mínima é 0,983 sem o viés inicial e 1,000 com ele. Com
  `fleet_utilization` de 0,95 ainda havia até 33% de partículas inviáveis
  (falha de empacotamento) e com 0,85 ou menos voltavam instâncias sem solução,
  por isso o padrão é 0,9.
- **Diversidade (S4).** Os perfis de entrega distintos passam de 1 (sempre) para
  entre 24 e 900 ao longo da busca, e sortear os genes de `Q` passa a mudar o
  custo. A população ainda converge: 24 perfis na iteração 600.
- **Qualidade.** Na S4, com 5 sementes e 1.000 iterações, o decodificador
  anterior (`just_in_time`, sem viés) tem custo médio `0,327` e o final `0,335`,
  sem diferença: o custo cai em dois patamares (~0,30 e ~0,34) que dependem da
  semente. Em 12 instâncias de 20 clientes (uma semente) a média foi `0,274`
  contra `0,258` e o uso de rotas caiu de 40,9 para 37,8, mas as diferenças por
  instância estão no nível do ruído de semente. Em instâncias folgadas o
  ganho é de viabilidade e diversidade, não de custo.
- **MIP completo na S4** (5 minutos, 2 threads, `aggregate_vehicles`): solução
  inicial `0,3442`, objetivo final `0,2489`, bound `0,1656`, gap `33%`, equivalente
  aos `0,2405` e `30%` do decodificador anterior.

**Limitações.** Só o gene `(p=0, v=0)` de cada `(cliente, período)` é lido; os
demais genes de `Q` continuam sem efeito e a mutação de `Q` sorteia entre todos
eles. O nivelamento garante a capacidade agregada da frota, não o empacotamento
exato: ainda pode haver partículas inviáveis em instâncias muito apertadas.

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


## Inicialização pela matheurística

O PSO aceita uma solução completa opcional em `initial_solution`. O fluxo configurável,
os limites separados, a avaliação mono/multiobjetivo e as saídas complementares
são descritos em [matheuristic.md](matheuristic.md). A partícula inicial preserva rotas,
atribuições e quantidades fracionárias; as demais partículas são construídas normalmente.
`solver.pso.time_limit=null` mantém a parada por iterações; um limite positivo
interrompe entre iterações e conserva a melhor solução. Medições anteriores
registradas acima precedem a migração para quantidades contínuas e não constituem
avaliação de desempenho desta implementação.
