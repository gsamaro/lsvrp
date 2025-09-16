#################################################################################################
# Multi Product Prodction Routing Problem Greedy Constructive Heuristic
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

class MultProductProdctionRoutingProblemGreedyConstructiveHeuristic:

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
        self.variables={}
        self.solution={}
        self.mitStart = False
        self.solverGurobi = 0

    def setMitStart(self,mitStart):
        self.mitStart = mitStart

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
        solucao_t_i_p = [[[{'cliente': i,'produto': p,'periodo': t,'estoque': 0,'demanda': 0,'producaco': 0} for p in range(self.p)] for i in range(self.i)] for t in range(self.t)]
        estoque_i_p = [[self.I_p_i_0[p][i] for p in range(self.p)] for i in range(self.i)]
        capacities = [[self.C] for _ in range(self.v)]
        routes = []

        for t in range(self.t):
            capacidade_producao_restante = int(self.B)
            capacidade_veiculo_totais = [self.C for _ in range(self.v)]

            # pdb.set_trace()
            candidatos = []
            veiculo_corrente=0
            for i in range(1, self.i):
                for p in range(self.p):
                    demanda_t = self.d_p_i_t[p][i-1][t]
                    qte = 0

                    if(estoque_i_p[i][p] < demanda_t): 
                        produto_faltante = demanda_t - estoque_i_p[i][p]
                        disponibilidade_estoque = self.U_p_i[p][i] - estoque_i_p[i][p]
                        qte = min(produto_faltante, capacidade_producao_restante, disponibilidade_estoque)

                        capacidade_producao_restante -=qte
                        capacidade_veiculo_totais[veiculo_corrente] -= qte

                        if(capacidade_veiculo_totais[veiculo_corrente] < 0):

                            capacidade_veiculo_totais[veiculo_corrente] += qte
                            veiculo_corrente+=1
                            capacidade_veiculo_totais[veiculo_corrente] -= qte

                            if(veiculo_corrente>self.v):
                                self.log.error("modelo Inviavel pelas restrições de capacidade do veiculo")

                        if(capacidade_producao_restante<0):
                            self.log.error("modelo Inviavel pelas restrições de produção")
                            break


                    estoque_i_p[i][p] += qte
                    solucao_t_i_p[t][i][p]['producaco']+= qte
                    solucao_t_i_p[t][i][p]['cliente'] = i
                    solucao_t_i_p[t][i][p]['produto'] = p
                    solucao_t_i_p[t][i][p]['periodo'] = t
                    solucao_t_i_p[t][i][p]['estoque'] = estoque_i_p[i][p]
                    solucao_t_i_p[t][i][p]['demanda'] = demanda_t
                        
                    if(estoque_i_p[i][p]<self.U_p_i[p][i] and capacidade_producao_restante > 0 and capacidade_veiculo_totais[veiculo_corrente] > 0):
                        demanda_futura = sum(self.d_p_i_t[p][i-1][t+1:self.t])
                        custo_unitario = self.s_p[p] + self.c_p[p] + self.h_p_i[p][i]*demanda_futura
                        candidatos.append((i,p,'barato',1/custo_unitario))

            candidatos.sort(key=lambda x: x[3], reverse=True)

            iter=0
            positivos = [v for v in capacidade_veiculo_totais if v > 0]
            if positivos:
                maior_valor = max(positivos)
                veiculo_corrente = capacidade_veiculo_totais.index(maior_valor)

            while ( (capacidade_producao_restante > 0 and len(candidatos)!=0 and capacidade_veiculo_totais[veiculo_corrente] > 0) and iter<=len(candidatos)):
                top_k = math.ceil(self.alfa * len(candidatos))
                RCL = candidatos[:top_k]
                i,_,_,_ = random.choice(RCL)

                total_p = []
                for p in range(self.p):
                    demanda_futura = sum(self.d_p_i_t[p][i-1][t+1:self.t])
                    disponibilidade_estoque = self.U_p_i[p][i] - estoque_i_p[i][p]

                    faltante = demanda_futura - estoque_i_p[i][p]
                    if(faltante<0):
                        faltante = 0

                    total_p.append(min(capacidade_producao_restante, demanda_futura, disponibilidade_estoque, faltante, capacidade_veiculo_totais[veiculo_corrente]))


                capacidade_producao_restante -= sum(total_p)
                capacidade_veiculo_totais[veiculo_corrente] -= sum(total_p)

                if(capacidade_producao_restante<0 or capacidade_veiculo_totais[veiculo_corrente]< 0 ):
                    self.log.warning(f"Produção está negátiva: {capacidade_producao_restante} ou capcidade_veiculo negativo:{capacidade_veiculo_totais[veiculo_corrente]}")
                    capacidade_producao_restante += sum(total_p)
                    capacidade_veiculo_totais[veiculo_corrente] += sum(total_p)
                    iter+=1
                    break
                    
                else:
                    for p in range(len(total_p)):
                        estoque_i_p[i][p] += total_p[p]
                        solucao_t_i_p[t][i][p]['producaco']+= total_p[p]
                        solucao_t_i_p[t][i][p]['cliente'] = i
                        solucao_t_i_p[t][i][p]['produto'] = p
                        solucao_t_i_p[t][i][p]['periodo'] = t
                        solucao_t_i_p[t][i][p]['estoque'] = estoque_i_p[i][p]
                        solucao_t_i_p[t][i][p]['demanda'] = self.d_p_i_t[p][i-1][t]
                        candidatos = [c for c in candidatos if not (c[0]==i and c[1]==p)]

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

            # pdb.set_trace()
            candidates_t = list(dict.fromkeys(candidates_t))
            dem_t = [v for v in dem_t if v]
            D = self.getDistancesInPeriod(candidates_t)

            route, distance, demandas = self.greedyRoute.greedyRandomizedConstruction(candidates_t, dem_t, capacities, D, int(self.v), self.alfa, random.Random(self.seed))
            routes.append({'periodo':t ,'route':route,'distance':distance,'demandas':demandas})

        self.solution= {
            "production": solucao_t_i_p,
            "routes": routes
        }
        #self.log.info(json.dumps(final_solution, indent=4))

    def convertVariables(self):

        final_solution = self.solution
        Z = np.zeros((self.v,self.i,self.k,self.t), dtype=int)

        '''𝑧𝑣𝑖𝑘𝑡'''
        for t in range(len(final_solution["routes"])):
            for v in range(len(final_solution["routes"][t]["route"])):
                for i in range( len(final_solution["routes"][t]["route"][v])):
                    origem = final_solution["routes"][t]["route"][v][i]
                    if(i+1 == len(final_solution["routes"][t]["route"][v])):
                        destino = 0
                    else:
                        destino = final_solution["routes"][t]["route"][v][i+1]
                    Z[v, origem,destino, t] = 1
        R = np.zeros((self.p,self.v,self.i,self.k,self.t), dtype=int)
        Q = np.zeros((self.p,self.v,self.i,self.t), dtype=int)

        '''𝑟𝑝𝑣𝑖𝑘𝑡'''
        '''𝑞𝑝𝑣𝑖𝑡'''
        for t in range(len(final_solution["routes"])):
            for v in range(len(final_solution["routes"][t]["demandas"])):
                for i in range( len(final_solution["routes"][t]["demandas"][v]['entregas'])):
                    origem_i = final_solution["routes"][t]["demandas"][v]["entregas"][i]["cliente"]
                    if(i+1 == len(final_solution["routes"][t]["route"][v])):
                        destino_j = 0
                    else:
                        destino_j = final_solution["routes"][t]["demandas"][v]["entregas"][i+1]["cliente"]
                    for p in range(len(final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"])):
                        R[p,v,origem_i,destino_j,t] = final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"][p]["restante_veiculo"]
                for i in range( len(final_solution["routes"][t]["demandas"][v]['entregas'])):
                    origem_i = final_solution["routes"][t]["demandas"][v]["entregas"][i]["cliente"]
                    for p in range(len(final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"])):
                        Q[p,v,origem_i,t] = final_solution["routes"][t]["demandas"][v]["entregas"][i]["produtos"][p]["qte_entregue"]

        X = np.zeros((self.p,self.t), dtype=int)
        Y = np.zeros((self.p,self.t), dtype=int)
        I = np.zeros((self.p,self.i,self.t), dtype=int)
        '''𝐼𝑝𝑖𝑡'''
        for t in range(len(final_solution["production"])):
            producao = np.zeros((self.p), dtype=int)
            for i in range(len(final_solution["production"][t])):
                for p in range(len(final_solution["production"][t][i])):
                    producao[p]+=final_solution["production"][t][i][p]["producaco"]
                    I[p,i,t] = final_solution["production"][t][i][p]["estoque"] - final_solution["production"][t][i][p]["demanda"]


            for p in range(len(final_solution["production"][t][i])):
                if(producao[p]>0):
                    Y[p,t] = 1
                X[p,t] = producao[p]

        self.variables={"X":X, "Y":Y, "I":I, "Q":Q, "R":R, "Z":Z}

        return Z,X,Y,I,R,Q
    
    def getResultsSolverHeurisct(self):
        z=self.variables["Z"]
        x=self.variables["X"]
        y=self.variables["Y"]
        ii=self.variables["I"]
        r=self.variables["R"]
        q=self.variables["Q"]
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

    def solver(self,numThreads=None,timeLimit=None):
        self.construirSolucao()
        self.convertVariables()
        if(self.mitStart==True):
            self.solverGurobi = MPPRP(self.data,self.dir,self.log,{"start":True, "variables":self.variables})
            self.solverGurobi.solver(timeLimit=timeLimit,numThreads=numThreads)
           
    
    def getResults(self):
        if(self.mitStart==True):
            return self.solverGurobi.getResults()

        return self.getResultsSolverHeurisct()


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
   