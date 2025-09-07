#################################################################################################
# Multi Product Prodction Routing Problem
# Copyright 2024 Mateus Chacon

# Este programa é um software livre, você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU como publicada pela Fundação do Software Livre (FSF),
# na versão 3 da Licença, ou (a seu critério) qualquer versão posterior.

# Este programa é distribuído na esperança de que possa ser útil, mas SEM NENHUMA GARANTIA,
# e sem uma garantia implícita de ADEQUAÇÃO a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR.

# Veja a Licença Pública Geral GNU para mais detalhes
#################################################################################################
import gurobipy as gp
from gurobipy import GRB
from src.log.Logger import Logger
import time

class MultProductProdctionRoutingProblem:

    def __init__(self,map,dir,log:Logger,start):
        self.model = gp.Model("Multi_Product_Prodction_Routing_Problem") 
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
        self.relaxedModelObjVal = 0
        self.objBound = 0
        self.nodeCount = 0
        self.log:Logger = log
        self.start = start

    def createDecisionVariables(self):
        for p in range(self.p):
            for t in range(self.t):
                self.X_p_t[p,t] = self.model.addVar(vtype=GRB.INTEGER, name=f"X[{p},{t}]")
                self.Y_p_t[p,t] = self.model.addVar(vtype=GRB.BINARY, name=f"Y[{p},{t}]")
            for i in range(self.i):
                for t in range(self.t):
                    self.I_p_i_t[p,i,t] = self.model.addVar(vtype=GRB.INTEGER, name=f"I[{p},{i},{t}]")
            for v in range(self.v):
                for i in range(self.i):
                    for k in range(self.k):
                        for t in range(self.t):
                            self.R_p_v_i_k_t[p,v,i,k,t] = self.model.addVar(vtype=GRB.INTEGER, name=f"R[{p},{v},{i},{k},{t}]")
                for i in range(self.i):
                    for t in range(self.t):
                        self.Q_p_v_i_t[p,v,i,t] = self.model.addVar(vtype=GRB.INTEGER, name=f"Q[{p},{v},{i},{t}]")
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    for t in range(self.t):
                        self.Z_v_i_k_t[v,i,k,t] = self.model.addVar(vtype=GRB.BINARY, name=f"Z[{v},{i},{k},{t}]")

    def startVariables(self):


        print(f"{-self.start["variables"]["I"][0][1][0]}+{self.start["variables"]["Q"][0][0][1][0]} = {-self.start["variables"]["I"][0][1][0]+self.start["variables"]["Q"][0][0][1][0]} == -123"   )

        '''-I[0,1,0] + Q[0,0,1,0] = -123 -163+58 =-123'''
        # print(self.start["variables"]["X"][p][t])

        for p in range(self.p):
            for t in range(self.t):
                self.X_p_t[p,t].start=self.start["variables"]["X"][p][t]
                self.Y_p_t[p,t].start=self.start["variables"]["Y"][p][t]
            for i in range(self.i):
                for t in range(self.t):
                    self.I_p_i_t[p,i,t].start=self.start["variables"]["I"][p][i][t]
            for v in range(self.v):
                for i in range(self.i):
                    for k in range(self.k):
                        for t in range(self.t):
                            self.R_p_v_i_k_t[p,v,i,k,t].start=self.start["variables"]["R"][p][v][i][k][t]
                for i in range(self.i):
                    for t in range(self.t):
                        self.Q_p_v_i_t[p,v,i,t].start=self.start["variables"]["Q"][p][v][i][t]
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    for t in range(self.t):
                        self.Z_v_i_k_t[v,i,k,t].start=self.start["variables"]["Z"][v][i][k][t]

    def crateObjectiveFunction(self):
        objExpr_1 = gp.LinExpr()
        for p in range(self.p):
            for t in range(self.t):
                objExpr_1 += self.s_p[p] * self.Y_p_t[p,t] + self.c_p[p] * self.X_p_t[p,t]

        objExpr_2 = gp.LinExpr()
        for p in range(self.p):
            for i in range(self.i):
                for t in range(self.t):
                    objExpr_2+=self.h_p_i[p][i]*self.I_p_i_t[p,i,t]

        objExpr_3 = gp.LinExpr()
        for v in range(self.v):
            for k in range(1,self.k):
                for t in range(self.t):
                    objExpr_3+=self.f*self.Z_v_i_k_t[v,0,k,t]

        objExpr_4 = gp.LinExpr()
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    if(i!=k):
                        for t in range(self.t):
                            objExpr_4+=self.a_i_k[i][k]*self.Z_v_i_k_t[v,i,k,t]

        objExpr = gp.LinExpr()
        objExpr = objExpr_1 + objExpr_2 + objExpr_3 + objExpr_4
        self.model.setObjective(objExpr, GRB.MINIMIZE)
    
    def createEstablishInvetoryBalanceAtPlant(self):
        for p in range(self.p):
            for t in range(self.t):
                r1=0
                for v in range(self.v):
                    for i in range(1,self.i):
                        r1+=self.Q_p_v_i_t[p,v,i,t]
                if(t == 0 ):
                    self.model.addConstr(self.X_p_t[p,t]+self.I_p_i_0[p][0] - r1 == self.I_p_i_t[p,0,t], name=f"EQ_(2)_p={p}_t={t}")
                else:
                    self.model.addConstr(self.X_p_t[p,t]+self.I_p_i_t[p,0,t-1]-r1 == self.I_p_i_t[p,0,t], name=f"EQ_(2)_p={p}_t={t}")
                         
    def creteInventoryBalancingInventoryCustomers(self):  
        for p in range(self.p):
            for i in range(1,self.i):
                for t in range(self.t):
                    r2=0
                    for v in range(self.v):
                        r2+=self.Q_p_v_i_t[p,v,i,t]
                    if(t == 0 ):
                        self.model.addConstr(r2+self.I_p_i_0[p][i] - self.d_p_i_t[p][i-1][t]==self.I_p_i_t[p,i,t], name=f"EQ_(3)_p={p}_i={i}_t={t}")
                    else:
                        self.model.addConstr(r2+self.I_p_i_t[p,i,t-1]-self.d_p_i_t[p][i-1][t]==self.I_p_i_t[p,i,t], name=f"EQ_(3)_p={p}_i={i}_t={t}")

    def createPlantsMaximum(self):
        for t in range(self.t):
            r3 = 0
            for p in range(self.p): 
                r3+=self.b_p[p]*self.X_p_t[p,t]  
            self.model.addConstr(r3<=self.B, name=f"EQ_(4)_t={t+1}")

    def createRelationshipBetweenProduction(self):
        for p in range(self.p):
            for t in range(self.t):
                self.model.addConstr(self.X_p_t[p,t]<=self.M*self.Y_p_t[p,t], name=f"EQ_(5)_p={p}_t={t}")

    def createDelimitMaximumCapacityItemsAtPlant(self):
        for p in range(self.p):
            for i in range(self.i):
                for t in range(self.t):
                    self.model.addConstr(self.I_p_i_t[p,i,t]<=self.U_p_i[p][i], name=f"EQ_(6)_p={p}_i={i}_t={t}")

    def createVehiclePreventTransshipmentIntermediateNodes(self):
        for p in range(self.p):
            for v in range(self.v):
                for k in range(1,self.k): 
                    for t in range(self.t):
                        r7_a=0
                        r7_b=0
                        for i in range(self.i): 
                            if(k!=i):
                                r7_a+=self.R_p_v_i_k_t[p,v,i,k,t]
                        for l in range(self.i):
                            if(k!=l):
                                r7_b+=self.R_p_v_i_k_t[p,v,k,l,t]
                        self.model.addConstr(r7_a-r7_b==self.Q_p_v_i_t[p,v,k,t], name=f"EQ_(7)_p={p}_v={v}_k={k}_t={t}")
        
    def createEliminationSubroutes(self):
        for p in range(self.p):
            for t in range(self.t):
                r8_a=0
                r8_b=0
                r8_c=0
                for v in range(self.v):
                    for k in range(1,self.k): 
                        r8_a+=self.R_p_v_i_k_t[p,v,0,k,t]
                    for i in range(1,self.i): 
                        r8_b+=self.R_p_v_i_k_t[p,v,i,0,t]
                    for l in range(1,self.i):
                        r8_c+=self.Q_p_v_i_t[p,v,l,t]
                self.model.addConstr(r8_a-r8_b==r8_c, name=f"EQ_(8)_p={p}_t={t}")

    def createVehicleLoadCapacityDelimited(self):
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    for t in range(self.t):
                        if(i!=k):
                            r9=0
                            for p in range(self.p):
                                r9+=self.R_p_v_i_k_t[p,v,i,k,t]
                            self.model.addConstr(r9<=self.C*self.Z_v_i_k_t[v,i,k,t], name=f"EQ_(9)_v={v}_i={i}_k={k}_t={t}")

    def createImposeMostOneRouteEachVehicle(self):
        for v in range(self.v):
            for t in range(self.t):
                r10=0
                for k in range(1,self.k):
                    r10+=self.Z_v_i_k_t[v,0,k,t]
                self.model.addConstr(r10<=1, name=f"EQ_(10)_v={v}_t={t}")        

    def createEnsureRoutesOnlyPlant(self):
        for v in range(self.v):
            for k in range(self.k):
                for t in range(self.t):
                    r11_a=0
                    r11_b=0
                    for i in range(self.i): 
                        if(k!=i):
                            r11_a+=self.Z_v_i_k_t[v,i,k,t]
                    for l in range(self.i):
                        if(k!=l):
                            r11_b+=self.Z_v_i_k_t[v,k,l,t]
                    self.model.addConstr(r11_a-r11_b==0, name=f"EQ_(11)_v={v}_k={k}_t={t}")  

    def createVehicleMostVisitCustomerEachPeriod(self):
        for k in range(1,self.k):
            for t in range(self.t):
                r12=0
                for v in range(self.v):
                    for i in range(self.i): 
                        if(k!=i):
                            r12+=self.Z_v_i_k_t[v,i,k,t]
                self.model.addConstr(r12<=1, name=f"EQ_(12)_k={k}_t={t}")  

    def outModel(self):
        self.model.Params.OutputFlag = 1
        self.model.write(f"{self.dir}modelo.lp")

    def getResults(self):
        if(self.solCount==0):
            return [],[],[],[],[],[],0,0,self.time,self.solCount,self.relaxedModelObjVal,self.nodeCount,self.objBound
        
        '''self.log.info("*******************************")
        self.log.info("============ Z ================")
        self.log.info("*******************************")'''
        Z=[]
        for t in range(self.t):
            #self.log.info(f"\n\n============ periodo {t} ============")
            v_list =[]
            for v in range(self.v):
                self.log.info(f"\n============ veiculo {v} ============")
                i_list =[]
                for i in range(self.i):
                    k_list=[]
                    for k in range(self.k):
                        variable = abs(self.Z_v_i_k_t[v,i,k,t].x)
                        #self.log.info(f" origem: {i} destino: {k} == {variable}")
                        k_list.append(variable)
                    i_list.append(k_list)
                v_list.append(i_list)
            Z.append(v_list)
        #self.log.info("\n\n===============================\n\n")

        for t in range(len(Z)):
            #self.log.info(f"\n\n============ periodo {t} ============")
            v_list =[]
            for v in range(len(Z[t])):
                #self.log.info(f"\n============ veiculo {v} ============")
                for i in range(len(Z[t][v])):
                    string = ""
                    for k in range(len(Z[t][v][i])):
                        string+= str(Z[t][v][i][k]) + "\t"
                    #self.log.info(string)

        #self.log.info("*******************************")
        #self.log.info("============ Y ================")
        #self.log.info("*******************************")
        Y = []
        for t in range(self.t):
            self.log.info(f"\n\n============ periodo {t} ============")
            p_list_y=[]
            for p in range(self.p):
                variable = abs(self.Y_p_t[p,t].x)
                p_list_y.append(variable)
                # print("produto: ",p," == ", variable)
            Y.append(p_list_y)
        #self.log.info("\n\n===============================\n\n")

        #self.log.info("*******************************")
        #self.log.info("============ X ================")
        #self.log.info("*******************************")
        X = []
        for t in range(self.t):
            #self.log.info(f"\n\n============ periodo {t} ============")
            p_list_x=[]
            for p in range(self.p):
                p_list_x.append(self.X_p_t[p,t].x)
                # print("produto: ",p," == ", self.X_p_t[p,t].x)
            X.append(p_list_x)
        #self.log.info("\n\n===============================\n\n")

        #self.log.info("*******************************")
        #self.log.info("============ I ================")
        #self.log.info("*******************************")
        I=[]
        for t in range(self.t):
            # print("\n\n============ periodo ",t," ============")
            p_list_i=[]
            for i in range(self.i):
                i_list_i=[]
                # print("\n============ cliente ",i," ============")
                for p in range(self.p):
                    # print("produto: ",p," == ", self.I_p_i_t[p,i,t].x)
                    i_list_i.append(self.I_p_i_t[p,i,t].x)
                p_list_i.append(i_list_i)
            I.append(p_list_i)
        self.log.info("\n\n===============================\n\n")

        #self.log.info("*******************************")
        #self.log.info("============ R ================")
        #self.log.info("*******************************")
        R=[]
        for t in range(self.t):
            #self.log.info(f"\n\n============ periodo {t} ============")
            t_list=[]
            for v in range(self.v):
                #self.log.info(f"\n============ veiculo {v} ============")
                v_list=[]
                for p in range(self.p):
                    p_list=[]
                    for i in range(self.i):
                        i_list=[]
                        for k in range(self.k):
                            # print("\n============ cliente ",i," -> cliente ",k," ============")
                            i_list.append(float(self.R_p_v_i_k_t[p,v,i,k,t].x))
                            self.log.info(f"\nperiodo {t} -> veiculo {v} -> cliente {i} -> cliente {k} -> produto {p} == { float(self.R_p_v_i_k_t[p,v,i,k,t].x)}")
                            # print("produto: ",p," == ", self.R_p_v_i_k_t[p,v,i,k,t].x)
                        p_list.append(i_list)
                    v_list.append(p_list)
                t_list.append(v_list)
            R.append(t_list)
        #self.log.info("\n\n===============================\n\n")

        #self.log.info("*******************************")
        #self.log.info("============ Q ================")
        #self.log.info("*******************************")
        Q=[]
        for t in range(self.t):
            #self.log.info(f"\n\n============ periodo {t} ============")
            t_list=[]
            for v in range(self.v):
                #self.log.info(f"\n============ veiculo {v} ============")
                v_list=[]
                for p in range(self.p):
                    #self.log.info(f"\n============ cliente {i} ============")
                    p_list=[]
                    for i in range(self.i):
                        self.log.info(f"produto:{p} == {self.Q_p_v_i_t[p,v,i,t].x}")
                        p_list.append(self.Q_p_v_i_t[p,v,i,t].x)
                    v_list.append(p_list)
                t_list.append(v_list)
            Q.append(t_list)
        #self.log.info("\n\n===============================\n\n")
    
        return Z,X,Y,I,R,Q,self.model.ObjVal,self.model.MIPGap,self.time,self.solCount,self.relaxedModelObjVal,self.nodeCount,self.objBound

    def terminate(self):
        self.model.terminate()

    def generteRelax(self):
        relaxed = self.model.relax()
        relaxed.optimize()
        self.relaxedModelObjVal = relaxed.ObjVal

    def processInformationsSolver(self):
        self.solCount = self.model.SolCount
        self.objBound = self.model.ObjBound
        self.nodeCount = self.model.NodeCount

    def solver(self,numThreads=None,timeLimit=None):
        self.createDecisionVariables()

        self.log.info(f"Variabes.start == {self.start['start']}")
        if(self.start['start'] == True):
            self.startVariables()
        
        self.crateObjectiveFunction()
        self.createEstablishInvetoryBalanceAtPlant()
        self.creteInventoryBalancingInventoryCustomers()
        self.createPlantsMaximum()
        self.createRelationshipBetweenProduction()
        self.createDelimitMaximumCapacityItemsAtPlant()
        self.createVehiclePreventTransshipmentIntermediateNodes()
        self.createEliminationSubroutes()
        self.createVehicleLoadCapacityDelimited()
        self.createImposeMostOneRouteEachVehicle()
        self.createEnsureRoutesOnlyPlant()
        self.createVehicleMostVisitCustomerEachPeriod()
        self.outModel()
        self.generteRelax()

        if numThreads is not None:
            self.model.setParam("Threads", numThreads)
        if timeLimit is not None:
            self.model.setParam("TimeLimit", timeLimit)

        start_time = time.time()
        self.model.optimize()
        end_time = time.time()
        self.time = end_time - start_time

        self.processInformationsSolver()