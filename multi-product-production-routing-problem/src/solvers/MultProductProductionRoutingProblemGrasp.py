#################################################################################################
# Multi Product Prodction Routing Problem Gras
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
from src.log.Logger import Logger
from src.solvers.GreedyRandomizedConstructionRoute import GreedyRandomizedConstructionRoute as GR
from src.solvers.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP
from src.solvers.MultProductProdctionRoutingProblemGreedyConstructiveHeuristic import MultProductProdctionRoutingProblemGreedyConstructiveHeuristic as MPPRPG
import numpy as np
from typing import List

class MultProductProductionRoutingProblemGrasp:
    def __init__(self,map,dir,log:Logger,rng:np.random.Generator):
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
        self.rng=rng

    def setMitStart(self,mitStart):
        self.mitStart = mitStart

    def setMaxInter(self,max_inter):
        self.max_inter = max_inter
    
    def setAlfa(self,alfa):
        self.alfa = alfa
    
    def setSeed(self,seed):
        self.seed = seed


    # ==========================
    # Busca Local: 1-Move entre rotas
    # ==========================

    def total_demanda(self, solution: List[List[int]], demands: List[List[float]], c:List[int]) -> float:
        p_size = len(demands[0])
        vehicles = []
        #pdb.set_trace()
        for v in range(len(solution)):
            d = []
            d_t = [0.0 for _ in range(p_size)]
            # Somar demandas atendidas por veículo
            for i in range(len(solution[v])):
                client = solution[v][i]
                d_p_i = []
                for p in range(len(demands[client])):
                    d_t[p] += demands[client][p]
                    d_p_i.append(demands[client][p])
                d.append(d_p_i)
            
            # Atualizar demandas restantes
            d_e = []
            for d_i in range(len(d)):
                d_e_p = []
                for p in range(len(d[d_i])):
                    d_t[p] -= d[d_i][p]
                    d_e_p.append({'produto':p,'restante_veiculo':d_t[p], 'qte_entregue': d[d_i][p]})
                d_e.append({'cliente': c[v][d_i], 'produtos': d_e_p})

            vehicles.append({'veiculo':v, 'entregas': d_e})
        return vehicles

    def busca_local_2opt_uma_rota(self, rota, dist):
        melhor = rota
        melhor_custo = self.custo_rota(melhor, dist)
        melhorou = True

        while melhorou:
            melhorou = False
            for i in range(len(rota) - 1):
                for j in range(i + 2, len(rota) + 1):
                    nova = melhor[:i] + melhor[i:j][::-1] + melhor[j:]
                    novo_custo = self.custo_rota(nova, dist)
                    if novo_custo < melhor_custo:
                        melhor = nova
                        melhor_custo = novo_custo
                        melhorou = True
            rota = melhor
        return melhor

    def busca_local_2opt(self,rotas, dist):
        return [self.busca_local_2opt_uma_rota(r, dist) for r in rotas]


    def custo_rota(self, rota, dist):
        custo = 0
        atual = 0  # depósito
        for cliente in rota:
            custo += dist[atual][cliente]
            atual = cliente
        custo += dist[atual][0]  # retorno ao depósito
        return custo

    def custo_total(self, rotas, dist):
        return sum(self.custo_rota(r, dist) for r in rotas)
    
    def busca_local_1move(self,rotas, demandas, capacidade, dist):
        melhorou = True
        while melhorou:
            melhorou = False
            for i in range(len(rotas)):
                for j in range(len(rotas)):
                    if i == j:
                        continue
                    for ci in range(len(rotas[i])):
                        cliente = rotas[i][ci]
                        demanda_cliente = demandas[cliente]
                        if sum(demandas[c] for c in rotas[j]) + demanda_cliente <= capacidade:
                            nova_i = rotas[i][:ci] + rotas[i][ci+1:]
                            for cj in range(len(rotas[j]) + 1):
                                nova_j = rotas[j][:cj] + [cliente] + rotas[j][cj:]
                                novas_rotas = rotas[:]
                                novas_rotas[i] = nova_i
                                novas_rotas[j] = nova_j
                                custo_antigo = self.custo_total(rotas, dist)
                                custo_novo = self.custo_total(novas_rotas, dist)
                                if custo_novo < custo_antigo:
                                    rotas = novas_rotas
                                    melhorou = True
                                    break
                            if melhorou:
                                break
                    if melhorou:
                        break
                if melhorou:
                    break
        return rotas

    def process_busca_local_entre_rotas(self,demands,points,distancies,candidates):
        demand_product = []
        for i in range(len(demands)):
            demand_product.append(sum(demands[i]))

        routes_opt = self.busca_local_1move(points,demand_product,self.C, distancies)
        routes_opt = self.busca_local_2opt(routes_opt, distancies)

        routes = []
        for route in routes_opt:
            routes.append([candidates[i] for i in route])

        dist_total = self.custo_total(routes_opt,distancies), 
        demanda_total = self.total_demanda(routes_opt,demands,routes)

        return routes,dist_total[0],demanda_total,routes_opt


    # ==========================
    # ==========================
    # ==========================

    def solver(self,numThreads=None,timeLimit=None):

        print("chegou aqui! seed", self.seed)
        alpha = 0.2
        # for alpha in np.linspace(0, 1, 111):
        inst = MPPRPG(map=self.data,dir=self.dir,log=self.log,rng=self.rng)
        inst.solver()
        fo = inst.getValueObjectiveFunction()
        solution = inst.getSolution()
        demands = solution['demands']
        distancies = solution['distancies']
        points = solution['points']
        candidates=solution['candidates']

        routes_opt = []
        for t in range(len(solution['routes'])):

            # print(f"solution rotas: {solution['routes'][t]['route']}, distancia: {float(solution['routes'][t]['distance'])}")

            routes,dist_total,demanda_total,point = self.process_busca_local_entre_rotas(demands[t],points[t],distancies[t],candidates[t])
            routes_opt.append({'periodo':t ,'route':routes,'distance':dist_total,'demandas':demanda_total})

            # print(f"Novas rotas: {routes}, distancia: {dist_total} \n")


        solution['routes'] = routes_opt
        solution['points'] = point


        self.solution = solution


        self.convertVariables()
        if(self.mitStart==True):
            self.solverGurobi = MPPRP(self.data,self.dir,self.log,{"start":True, "variables":self.variables})
            self.solverGurobi.solver(timeLimit=timeLimit,numThreads=numThreads)

        print(f"FO: {self.getValueObjectiveFunction()}")


    # ==========================
    # Conversor de dados
    # ==========================
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
    
    def getValueObjectiveFunction(self):
        objExpr_1 = 0
        for p in range(self.p):
            for t in range(self.t):
                objExpr_1 += self.s_p[p] * self.variables["Y"][p][t] + self.c_p[p] * self.variables["X"][p][t]

        objExpr_2 = 0
        for p in range(self.p):
            for i in range(self.i):
                for t in range(self.t):
                    objExpr_2+=self.h_p_i[p][i]*self.variables["I"][p][i][t]

        objExpr_3 = 0
        for v in range(self.v):
            for k in range(1,self.k):
                for t in range(self.t):
                    objExpr_3+=self.f*self.variables["Z"][v][0][k][t]

        objExpr_4 = 0
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    if(i!=k):
                        for t in range(self.t):
                            objExpr_4+=self.a_i_k[i][k]*self.variables["Z"][v][i][k][t]

        return objExpr_1 + objExpr_2 + objExpr_3 + objExpr_4
    
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

    def getResults(self):
        if(self.mitStart==True):
            return self.solverGurobi.getResults()

        return self.getResultsSolverHeurisct()
    # ==========================
    # ==========================
    # ==========================