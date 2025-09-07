#################################################################################################
# Multi Product Prodction Routing Problem Grasp
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from src.log.Logger import Logger
import random
from typing import List
from src.solvers.GreedyRandomizedConstructionRoute import GreedyRandomizedConstructionRoute as GR
from src.helpers.ReadPrpFile import ReadPrpFile as RD
from src.solvers.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
import pdb
import numpy as np
import math
import json

class MultProductProdctionRoutingProblemGrasp:

    def __init__(self,map,dir,log:Logger):
        self.data = map
        self.p=map['num_products']            ##Products  
        self.i=map['num_customers'] + 1       ##Customers
        self.k=map['num_customers'] + 1       ##Customers
        self.t=map['num_periods']             ##Periods
        self.v=map['num_vehicles']            ##Vehicles         
        self.B=map['B']                       ##Production capacity;
        self.b_p=map['b_p']                   ##Time required to produce item 𝑝;
        self.c_p=map['c_p']                   ##Production cost of item 𝑝;
        self.s_p=map['s_p']                   ##Setup cost of item 𝑝;
        self.M=map['M']                       ##Big number 
        self.U_p_i=map['U_pi']                ##Maximum inventory upper bound of item 𝑝 at site i;
        self.I_p_i_0=map['I_pi0']             ##Initial Inventory of item 𝑝 at site 𝑖;
        self.h_p_i=map['h_pi']                ##Inventory cost of item 𝑝 at site 𝑖;
        self.C=map['C']                       ##Vehicle capacity;
        self.f=map['f']                       ##Fixed transportation cost;
        self.a_i_k=map['a_ik']                ##Transportation cost for traveling from node 𝑖 to node k;
        self.d_p_i_t=map['d_pit']             ##Demand of item 𝑝 at customer 𝑖 in period 𝑡.
        self.X_p_t={}                         ##Quantity of item 𝑝 produced in period 𝑡.
        self.Y_p_t={}                         ##1, if item 𝑝 is produced in period 𝑡; or 0, otherwise.
        self.I_p_i_t={}                       ##Inventory of item 𝑝 at site 𝑖 in the end of period 𝑡.
        self.Z_v_i_k_t={}                     ##1, if vehicle v travels along edge (i,k) in period t; or 0, atherwise.
        self.R_p_v_i_k_t={}                   ##Quantity of item 𝑝 transported by vehicle 𝑣 on edge (𝑖, 𝑘) in period 𝑡;
        self.Q_p_v_i_t={}                     ##Quantity of item 𝑝 delivered by vehicle 𝑣 to customer 𝑖 in period 𝑡.
        self.dir = dir
        self.time = 0
        self.solCount = 0
        self.log:Logger = log
        self.max_inter = 100
        self.alfa = 0.2
        self.seed = 123
        self.greedyRoute = GR(log=log)
        self.s = 0
        self.variables={}

    def setMaxInter(self,max_inter):
        self.max_inter = max_inter
    
    def setAlfa(self,alfa):
        self.alfa = alfa
    
    def setSeed(self,seed):
        self.seed = seed

    def getDistancesInPeriod(self,candidates):
        D = np.zeros((len(candidates), len(candidates)), dtype=float)
        for i in range(len(candidates)):
            for k in range(len(candidates)):
                d = self.a_i_k[candidates[i]][candidates[k]]
                D[i, k] = d
                D[k, i] = d
        return D


    def construirSolucao(self):
        """
        Constrói uma solução (produção + roteiros) respeitando capacidades de produção e veículos,
        e prepara os tensores X, Y, I, Q, R, Z para o modelo.
        Mantém a compatibilidade com o restante do código (chaves e formato de retorno).
        """
        # ---------- Iniciais ----------
        # produção/estoque por (t, i, p)
        solucao_t_i_p = [
            [
                [
                    {
                        "cliente": i,
                        "produto": p,
                        "periodo": t,
                        "estoque": 0,
                        "demanda": 0,
                        "producaco": 0,
                    }
                    for p in range(self.p)
                ]
                for i in range(self.i)
            ]
            for t in range(self.t)
        ]

        # estoque inicial I_{p,i,0} -> organização [i][p]
        estoque_i_p = [[self.I_p_i_0[p][i] for p in range(self.p)] for i in range(self.i)]

        # capacidade de cada veículo (para passar ao construtor de rotas)
        capacities = [[self.C] for _ in range(self.v)]

        routes = []

        # ---------- Helpers internos ----------
        def demanda_futura(p, i, t_atual):
            """Soma de demanda futura para (p,i) de t_atual+1 até T-1."""
            if t_atual + 1 >= self.t:
                return 0
            return sum(self.d_p_i_t[p][i - 1][t_atual + 1 : self.t])

        def escolher_veiculo_maior_cap(restantes):
            """Retorna índice do veículo com maior capacidade restante; se tudo zerado, 0."""
            positivos = [v for v in restantes if v > 0]
            if not positivos:
                return 0
            m = max(positivos)
            return restantes.index(m)

        # ---------- Laço principal por período ----------
        for t in range(self.t):
            cap_producao_rest = int(self.B)
            cap_veic_rest = [self.C for _ in range(self.v)]
            veic_atual = 0

            candidatos = []  # cada item: (i, score)
            # 1) Atender demanda corrente (produção mínima necessária)
            for i in range(1, self.i):  # cliente 0 costuma ser o depósito
                for p in range(self.p):
                    dem_t = self.d_p_i_t[p][i - 1][t]
                    qte = 0

                    if estoque_i_p[i][p] < dem_t:
                        faltante = dem_t - estoque_i_p[i][p]
                        disp_estoque = self.U_p_i[p][i] - estoque_i_p[i][p]
                        qte = min(faltante, cap_producao_rest, disp_estoque)

                        # tenta usar veículo corrente; se estourar, passa pro próximo
                        if qte > 0:
                            cap_veic_rest[veic_atual] -= qte
                            if cap_veic_rest[veic_atual] < 0:
                                # desfaz e tenta próximo veículo
                                cap_veic_rest[veic_atual] += qte
                                veic_atual += 1
                                if veic_atual >= self.v:
                                    self.log.error("Modelo inviável: sem capacidade veicular.")
                                    veic_atual = self.v - 1
                                    qte = 0  # não produz (evita índice fora do limite)
                                else:
                                    cap_veic_rest[veic_atual] -= qte

                            cap_producao_rest -= qte
                            if cap_producao_rest < 0:
                                self.log.error("Modelo inviável: capacidade de produção excedida.")
                                # desfaz a produção aplicada
                                cap_producao_rest += qte
                                cap_veic_rest[veic_atual] += qte
                                qte = 0

                    # aplica efeitos (mesmo se qte = 0, atualiza metadados)
                    estoque_i_p[i][p] += qte
                    cel = solucao_t_i_p[t][i][p]
                    cel["producaco"] += qte
                    cel["cliente"] = i
                    cel["produto"] = p
                    cel["periodo"] = t
                    cel["estoque"] = estoque_i_p[i][p]
                    cel["demanda"] = dem_t

                    # 2) Considerar produzir extra (estoque antecipado) se ainda houver capacidade
                    if (
                        estoque_i_p[i][p] < self.U_p_i[p][i]
                        and cap_producao_rest > 0
                        and cap_veic_rest[veic_atual] > 0
                    ):
                        dem_fut = demanda_futura(p, i, t)
                        # custo efetivo p/ priorização: (s + c + h * demanda_futura)
                        custo_unit = self.s_p[p] + self.c_p[p] + self.h_p_i[p][i] * dem_fut
                        if custo_unit > 0:
                            # candidato por cliente (i) – usamos 1/custo como score para ordenar
                            candidatos.append((i, 1.0 / custo_unit))

            # ordena candidatos por "barateza"
            # se o cliente aparecer mais de uma vez (por produtos diferentes), vamos unificar depois
            candidatos.sort(key=lambda x: x[1], reverse=True)

            # usa o veículo com maior capacidade restante
            veic_atual = escolher_veiculo_maior_cap(cap_veic_rest)

            # 3) Produção adicional (estoque) guiada por RCL por cliente
            # Consolidar candidatos por cliente (melhor score)
            score_por_cliente = {}
            for i, s in candidatos:
                score_por_cliente[i] = max(s, score_por_cliente.get(i, -float("inf")))
            candidatos_unicos = sorted(score_por_cliente.items(), key=lambda x: x[1], reverse=True)

            iter_count = 0
            while (
                cap_producao_rest > 0
                and candidatos_unicos
                and cap_veic_rest[veic_atual] > 0
                and iter_count <= len(candidatos_unicos)
            ):
                top_k = max(1, math.ceil(self.alfa * len(candidatos_unicos)))
                RCL = candidatos_unicos[:top_k]
                i, _ = random.choice(RCL)

                # decide quanto produzir, por produto, para estoque futuro
                total_p = [0] * self.p
                for p in range(self.p):
                    dem_fut = demanda_futura(p, i, t)
                    disp_estoque = self.U_p_i[p][i] - estoque_i_p[i][p]
                    faltante_fut = max(0, dem_fut - estoque_i_p[i][p])

                    q = min(cap_producao_rest, dem_fut, disp_estoque, faltante_fut, cap_veic_rest[veic_atual])
                    total_p[p] = max(0, int(q))  # garantir inteiro não-negativo

                soma = sum(total_p)

                if soma <= 0:
                    # remove cliente da lista e continua
                    candidatos_unicos = [c for c in candidatos_unicos if c[0] != i]
                    iter_count += 1
                    continue

                # aplica consumo de capacidades
                cap_producao_rest -= soma
                cap_veic_rest[veic_atual] -= soma

                if cap_producao_rest < 0 or cap_veic_rest[veic_atual] < 0:
                    # desfaz e encerra iteração
                    self.log.warning(
                        f"Capacidades negativas detectadas: prod={cap_producao_rest}, veic={cap_veic_rest[veic_atual]}"
                    )
                    cap_producao_rest += soma
                    cap_veic_rest[veic_atual] += soma
                    iter_count += 1
                    break

                # aplica produção e atualiza células
                for p in range(self.p):
                    q = total_p[p]
                    if q <= 0:
                        continue
                    estoque_i_p[i][p] += q
                    cel = solucao_t_i_p[t][i][p]
                    cel["producaco"] += q
                    cel["cliente"] = i
                    cel["produto"] = p
                    cel["periodo"] = t
                    cel["estoque"] = estoque_i_p[i][p]
                    cel["demanda"] = self.d_p_i_t[p][i - 1][t]

                # remove o cliente escolhido (já atendido nesta rodada)
                candidatos_unicos = [c for c in candidatos_unicos if c[0] != i]
                # re-escolhe veículo com maior capacidade restante
                veic_atual = escolher_veiculo_maior_cap(cap_veic_rest)
                iter_count += 1

            # 4) Montagem dos candidatos de roteamento e demandas agregadas por cliente no período t
            candidates_t = [0]  # 0 = depósito
            dem_t = [[0.0] * self.p]  # vetor de produtos no depósito (zerado)

            for i, linha in enumerate(solucao_t_i_p[t]):
                prod = []
                client_current = 0
                for p, celula in enumerate(linha):
                    prod.append(celula['producaco'])  # já é separado por produto
                    estoque_i_p[i][p] = celula['estoque'] - celula['demanda']
                    client_current = celula['cliente']

                if any(q > 0 for q in prod):  # se tem algo produzido para o cliente
                    candidates_t.append(client_current)
                    dem_t.append(prod)  # agora vai com vetor por produto, não o somatório

            # remove duplicados em candidates_t mantendo ordem
            candidates_t = list(dict.fromkeys(candidates_t))
            # mantém dem_t coerente (primeiro elemento é o depósito 0 com 0.0)
            dem_t = [row for row in dem_t if row is not None]

            D = self.getDistancesInPeriod(candidates_t)

            # chamada do construtor de rotas (precisa devolver (route, distance, demandas))
            route_distance_dem = self.greedyRoute.greedyRandomizedConstruction(
                candidates_t, dem_t, capacities, D, int(self.v), self.alfa, random.Random(self.seed)
            )
            if not route_distance_dem or len(route_distance_dem) != 3:
                self.log.error("greedyRandomizedConstruction não retornou (route, distance, demandas).")
                route, distance, demandas = [], 0.0, []
            else:
                route, distance, demandas = route_distance_dem

            routes.append({"periodo": t, "route": route, "distance": distance, "demandas": demandas})

        # ---------- Empacotamento final ----------
        final_solution = {"production": solucao_t_i_p, "routes": routes}
        self.log.info(json.dumps(final_solution, indent=4))

        # Z[v,i,k,t] arestas percorridas
        Z = np.zeros((self.v, self.i, self.k, self.t), dtype=int)
        for t in range(len(final_solution["routes"])):
            rotas_t = final_solution["routes"][t]["route"]
            for v in range(len(rotas_t)):
                rota_v = rotas_t[v]
                for idx_i in range(len(rota_v)):
                    origem = rota_v[idx_i]
                    destino = rota_v[idx_i + 1] if (idx_i + 1) < len(rota_v) else 0
                    Z[v, origem, destino, t] = 1

        # R[p,v,i,k,t] e Q[p,v,i,t]
        R = np.zeros((self.p, self.v, self.i, self.k, self.t), dtype=int)
        Q = np.zeros((self.p, self.v, self.i, self.t), dtype=int)

        for t in range(len(final_solution["routes"])):
            dem_t_v = final_solution["routes"][t]["demandas"]
            rotas_t = final_solution["routes"][t]["route"]

            for v in range(len(dem_t_v)):
                entregas_v = dem_t_v[v].get("entregas", [])
                rota_v = rotas_t[v] if v < len(rotas_t) else []

                # R usa pares (origem->destino) na sequência da rota
                for i_idx in range(len(entregas_v)):
                    origem_i = entregas_v[i_idx]["cliente"]
                    proximo_cliente = (
                        entregas_v[i_idx + 1]["cliente"] if (i_idx + 1) < len(entregas_v) else 0
                    )
                    destino_j = proximo_cliente

                    for p in range(len(entregas_v[i_idx]["produtos"])):
                        R[p, v, origem_i, destino_j, t] = entregas_v[i_idx]["produtos"][p].get(
                            "restante_veiculo", 0
                        )

                # Q quantidade entregue por (p,v,i,t)
                for i_idx in range(len(entregas_v)):
                    origem_i = entregas_v[i_idx]["cliente"]
                    for p in range(len(entregas_v[i_idx]["produtos"])):
                        Q[p, v, origem_i, t] = entregas_v[i_idx]["produtos"][p].get(
                            "qte_entregue", 0
                        )

        # X[p,t], Y[p,t], I[p,i,t]
        X = np.zeros((self.p, self.t), dtype=int)
        Y = np.zeros((self.p, self.t), dtype=int)
        I = np.zeros((self.p, self.i, self.t), dtype=int)

        for t in range(len(final_solution["production"])):
            producao_p = np.zeros((self.p,), dtype=int)
            for i in range(len(final_solution["production"][t])):
                for p in range(self.p):
                    cel = final_solution["production"][t][i][p]
                    producao_p[p] += cel["producaco"]
                    I[p, i, t] = cel["estoque"] - cel["demanda"]

            for p in range(self.p):
                if producao_p[p] > 0:
                    Y[p, t] = 1
                X[p, t] = producao_p[p]

        self.variables = {"X": X, "Y": Y, "I": I, "Q": Q, "R": R, "Z": Z}

        # mantém a mesma assinatura/ordem de retorno usada por você
        return Z, X, Y, I, R, Q, 0, 0, 0, 0

    def getResultsSolver(self):

        z,x,y,ii,r,q,_,_,_,_ = self.s

        '''print("*******************************")
        print("============ Z ================")
        print("*******************************")'''
        Z=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            v_list =[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                i_list =[]
                for i in range(self.i):
                    k_list=[]
                    for k in range(self.k):
                        variable = z[v,i,k,t]
                        #print(" origem: ",i," destino: ",k," == ",variable)
                        k_list.append(int(variable))
                    i_list.append(k_list)
                v_list.append(i_list)
            Z.append(v_list)
        #print("\n\n===============================\n\n")

        '''for t in range(len(Z)):
            print("\n\n============ periodo ",t," ============")
            for v in range(len(Z[t])):
                print("\n============ veiculo ",v," ============")
                for i in range(len(Z[t][v])):
                    string = ""
                    for k in range(len(Z[t][v][i])):
                        string+= str(Z[t][v][i][k]) + "\t"
                    print(string)'''

        '''print("*******************************")
        print("============ Y ================")
        print("*******************************")'''
        Y = []
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_y=[]
            for p in range(self.p):
                variable = abs(y[p,t])
                p_list_y.append(int(variable))
                #print("produto: ",p," == ", variable)
            Y.append(p_list_y)
        #print("\n\n===============================\n\n")

        '''print("*******************************")
        print("============ X ================")
        print("*******************************")'''
        X = []
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_x=[]
            for p in range(self.p):
                p_list_x.append(int(x[p,t]))
                #print("produto: ",p," == ",x[p,t])
            X.append(p_list_x)
        '''print("\n\n===============================\n\n")

        print("*******************************")
        print("============ I ================")
        print("*******************************")'''
        I=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            p_list_i=[]
            for i in range(self.i):
                i_list_i=[]
                #print("\n============ cliente ",i," ============")
                for p in range(self.p):
                    #print("produto: ",p," == ", ii[p,i,t])
                    i_list_i.append(int(ii[p,i,t]))
                p_list_i.append(i_list_i)
            I.append(p_list_i)
        '''print("\n\n===============================\n\n")

        print("*******************************")
        print("============ R ================")
        print("*******************************")'''
        R=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            t_list=[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                v_list=[]
                for p in range(self.p):
                    p_list=[]
                    for i in range(self.i):
                        i_list=[]
                        for k in range(self.k):
                            #self.log.info(f"\nperiodo {t} -> veiculo {v} -> cliente {i} -> cliente {k} -> produto {p} == { r[p,v,i,k,t]}",)
                            #print("\n============ cliente ",i," -> cliente ",k," ============")
                            i_list.append(float(r[p,v,i,k,t]))
                            #print("produto: ",p," == ", r[p,v,i,k,t])
                        p_list.append(i_list)
                    v_list.append(p_list)
                t_list.append(v_list)
            R.append(t_list)
        '''print("\n\n===============================\n\n")

        print("*******************************")
        print("============ Q ================")
        print("*******************************")'''
        Q=[]
        for t in range(self.t):
            #print("\n\n============ periodo ",t," ============")
            t_list=[]
            for v in range(self.v):
                #print("\n============ veiculo ",v," ============")
                v_list=[]
                for p in range(self.p):
                    #print("\n============ cliente ",i," ============")
                    p_list=[]
                    for i in range(self.i):
                        #self.log.info(f"\nperiodo {t} -> veiculo {v} -> cliente -> {i} -> produto {p} == { q[p,v,i,t]}",)
                        #print("produto: ",p," == ",q[p,v,i,t])
                        p_list.append(int(q[p,v,i,t]))
                    v_list.append(p_list)
                t_list.append(v_list)
            Q.append(t_list)
        #print("\n\n===============================\n\n")
    
        return Z,X,Y,I,R,Q,0,0,0,0,0,0,0   

    def grasp(self,numThreads,timeLimit):

        melhor_solucao = {}
        self.s = self.construirSolucao()


        instance = MPPRP(self.data,self.dir,self.log,{"start":True, "variables":self.variables})
        instance.solver(timeLimit=timeLimit,numThreads=numThreads)
        Z,X,Y,I,R,Q,FO,GAP,TIME,SOL_COUNT, RELAXED_MODEL_OBJE_VAL, NODE_COUNT, OBJ_BOUND = instance.getResults()


        #for ite in range(self.max_inter):



            

        return 0


    def solver(self,numThreads=None,timeLimit=None):

        self.grasp(numThreads,timeLimit)
        return []
    
    def getResults(self):
        return self.getResultsSolver()


    """
    Busca_Local(solucao):
    melhora ← verdadeiro
    
    enquanto melhora faça:
        melhora ← falso
        melhor_movimento ← ∅
        melhor_custo ← custo(solucao)
        
        // Explorar vizinhanças
        Para cada cliente c ∈ C:
            Para cada produto p ∈ P:
                Para cada período t ∈ T:
                    
                    // Movimento 1: realocar produção para outro período
                    Para cada período t2 próximo de t:
                        nova_solucao ← mover_producao(solucao, c, p, t, t2)
                        se viável(nova_solucao):
                            custo ← Avaliar(nova_solucao)
                            se custo < melhor_custo:
                                melhor_movimento ← (c,p,t,t2)
                                melhor_custo ← custo
                        
                    // Movimento 2: redistribuir entre clientes
                    Para cada cliente c2 ≠ c:
                        nova_solucao ← transferir_producao(solucao, c → c2, p, t)
                        se viável(nova_solucao):
                            custo ← Avaliar(nova_solucao)
                            se custo < melhor_custo:
                                melhor_movimento ← (c,c2,p,t)
                                melhor_custo ← custo
                                
                    // Movimento 3: reduzir excesso de estoque
                    se estoque[c][p][t] >> demanda[c][p][t]:
                        nova_solucao ← reduzir_estoque(solucao, c,p,t)
                        se viável(nova_solucao):
                            custo ← Avaliar(nova_solucao)
                            se custo < melhor_custo:
                                melhor_movimento ← (reduzir,c,p,t)
                                melhor_custo ← custo
                                
        // Aplicar melhor movimento encontrado
        se melhor_movimento ≠ ∅:
            aplicar(melhor_movimento, solucao)
            melhora ← verdadeiro
    
    retornar solucao

    """
   