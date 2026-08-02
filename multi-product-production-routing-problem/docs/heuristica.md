# 3.3 Geração de soluções factíveis

## Bounds relaxados de dimensionamento para o PSO

Antes da inicialização do enxame, o PSO pode resolver o PL auxiliar de
dimensionamento implementado em `LotSizingRelaxation`. O modelo mantém apenas
as restrições 8, 9, 10 e 12 do modelo matemático, além da não negatividade,
considerando as variáveis contínuas de produção `X`, estoque `I` e entrega
`Q`. Roteamento, setup e capacidade dos veículos permanecem na etapa de
construção da solução factível.

Agora são resolvidos dois PLs globais: um minimizando `sum(X[p,t])` e outro
maximizando `sum(X[p,t])`. O retorno fornece os valores agregados `lower` e
`upper`, além das soluções completas dos dois PLs. As matrizes de produção
`lower_solution['X']` e `upper_solution['X']` são usadas como bounds
`LB[p,t]` e `UB[p,t]` na `FeasibleParticleHeuristic`; os escalares agregados
servem para avaliação do relaxamento. O segundo PL recebe restrições
`X[p,t] >= lower_solution['X'][p,t]`, garantindo que o perfil upper não fique
abaixo do perfil lower. A construção factível permanece ativa.

Ao construir cada período, a heurística primeiro entrega as quantidades
necessárias para cobrir os déficits dos clientes. Em seguida, distribui todo o
excedente disponível na planta para os clientes, respeitando a capacidade de
estoque `U[p][i]` de cada produto no cliente, a carga `C` do veículo que atende
o cliente e a capacidade total da frota. A capacidade de estoque da planta é
`U[p][0]`; como o excedente é escoado, o estoque final da planta é zero em cada
período. As preferências codificadas pelos genes de `Q` ordenam essa alocação
adicional, preservando perfis de entrega diversos entre as partículas. Se o
excedente não puder ser escoado sob essas capacidades, a partícula é inválida.

### Limitação em avaliação

Ainda são geradas partículas infactíveis. Isso ocorre quando o perfil de
produção imposto pelos bounds não pode ser escoado no período respeitando, ao
mesmo tempo, os limites de estoque `U[p][i]`, a carga `C` de cada veículo e a
alocação de clientes em rotas. Nesses casos a partícula é reamostrada; a
integração dos limites de entrega na geração dos bounds permanece pendente de
avaliação.

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
