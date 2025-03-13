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

class MultProductProdctionRoutingProblem:

    def __init__(self,map):
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
                    # if(i!=k):
                    for t in range(self.t):
                        objExpr_4+=self.a_i_k[i][k]*self.Z_v_i_k_t[v,i,k,t]

        objExpr = gp.LinExpr()
        objExpr = objExpr_1 + objExpr_2 + objExpr_3 + objExpr_4
        self.model.setObjective(objExpr, GRB.MINIMIZE)
    
    def createEstablishInvetoryBalanceAtPlant(self):
        for p in range(self.p):
            for i in range(self.i):
                self.model.addConstr(self.I_p_i_t[p,i,0]==self.I_p_i_0[p][i], name=f"[0] rest_{p}_{i}")

        for p in range(self.p):
            for t in range(1,self.t):
                for v in range(self.v):
                    r1=0
                    for i in range(1,self.i):
                        r1+=self.Q_p_v_i_t[p,v,i,t]
                self.model.addConstr(self.X_p_t[p,t]+self.I_p_i_t[p,0,t-1]-r1 == self.I_p_i_t[p,0,t], name=f"[1] rest_{p}_{t}")

    def creteInventoryBalancingInventoryCustomers(self):  
        for p in range(self.p):
            for i in range(1,self.i):
                for t in range(1,self.t):
                    r2=0
                    for v in range(self.v):
                        r2+=self.Q_p_v_i_t[p,v,i,t]
                    self.model.addConstr(r2+self.I_p_i_t[p,i,t-1]-self.d_p_i_t[p][i-1][t]==self.I_p_i_t[p,i,t], name=f"[2] rest_{p}_{i}_{t}")

    def createPlantsMaximum(self):
        for t in range(self.t):
            r3 = 0
            for p in range(self.p): 
                r3+=self.b_p[p]*self.X_p_t[p,t]  
            self.model.addConstr(r3<=self.B, name=f"[3] rest_{p}")

    def createRelationshipBetweenProduction(self):
        for p in range(self.p):
            for t in range(self.t):
                self.model.addConstr(self.X_p_t[p,t]<=self.M*self.Y_p_t[p,t], name=f"[4] rest_{p}_{t}")

    def createDelimitMaximumCapacityItemsAtPlant(self):
        for p in range(self.p):
            for i in range(self.i):
                for t in range(self.t):
                    self.model.addConstr(self.I_p_i_t[p,i,t]<=self.U_p_i[p][i], name=f"[5] rest_{p}_{i}_{t}")

    def createVehiclePreventTransshipmentIntermediateNodes(self):
        for p in range(self.p):
            for v in range(self.v):
                for k in range(1,self.k): 
                    for t in range(self.t):
                        r7_a=0
                        r7_b=0
                        for i in range(self.i): 
                            # if(k!=i):
                            r7_a+=self.R_p_v_i_k_t[p,v,i,k,t]
                        for l in range(self.i):
                            # if(k!=l):
                            r7_b+=self.R_p_v_i_k_t[p,v,k,l,t]
                        self.model.addConstr(r7_a-r7_b==self.Q_p_v_i_t[p,v,k,t], name=f"[6] rest_{p}_{v}_{k}_{t}")
        
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
                self.model.addConstr(r8_a-r8_b==r8_c, name=f"[7] rest_{p}_{t}")

    def createVehicleLoadCapacityDelimited(self):
        for v in range(self.v):
            for i in range(self.i):
                for k in range(self.k):
                    for t in range(self.t):
                        r9=0
                        for p in range(self.p):
                            r9+=self.R_p_v_i_k_t[p,v,i,k,t]
                        self.model.addConstr(r9<=self.C*self.Z_v_i_k_t[v,i,k,t], name=f"[8] rest_{v}_{i}_{k}_{t}")

    def createImposeMostOneRouteEachVehicle(self):
        for v in range(self.v):
            for t in range(self.t):
                r10=0
                for k in range(self.k):
                    r10+=self.Z_v_i_k_t[v,0,k,t]
                self.model.addConstr(r10<=1, name=f"[9] rest_{v}_{t}")        

    def createEnsureRoutesOnlyPlant(self):
        for v in range(self.v):
            for k in range(self.k):
                for t in range(self.t):
                    r11_a=0
                    r11_b=0
                    for i in range(self.i): 
                        # if(k!=i):
                        r11_a+=self.Z_v_i_k_t[v,i,k,t]
                    for l in range(self.i):
                        # if(k!=l):
                        r11_b+=self.Z_v_i_k_t[v,k,l,t]
                    self.model.addConstr(r11_a-r11_a==0, name=f"[10] rest_{v}_{k}_{t}")  

    def createVehicleMostVisitCustomerEachPeriod(self):
        for k in range(1,self.k):
            for t in range(self.t):
                r12=0
                for v in range(self.v):
                    for i in range(self.i): 
                        # if(k!=i):
                        r12+=self.Z_v_i_k_t[v,i,k,t]
                self.model.addConstr(r12<=1, name=f"[11] rest_{k}_{t}")  

    def outModel(self):
        self.model.Params.OutputFlag = 1
        self.model.write("./out/modelo.lp")

    def getResults(self):

        print("*******************************")
        print("============ Z ================")
        print("*******************************")
        Z=[]
        for t in range(self.t):
            print("\n\n============ periodo ",t," ============")
            v_list =[]
            for v in range(self.v):
                print("\n============ veiculo ",v," ============")
                i_list =[]
                for i in range(self.i):
                    k_list=[]
                    for k in range(self.k):
                        variable = abs(self.Z_v_i_k_t[v,i,k,t].x)
                        print(" origem: ",i," destino: ",k," == ",variable)
                        k_list.append(variable)
                    i_list.append(k_list)
                v_list.append(i_list)
            Z.append(v_list)
        print("\n\n===============================\n\n")

        for t in range(len(Z)):
            print("\n\n============ periodo ",t," ============")
            for v in range(len(Z[t])):
                print("\n============ veiculo ",v," ============")
                for i in range(len(Z[t][v])):
                    string = ""
                    for k in range(len(Z[t][v][i])):
                        string+= str(Z[t][v][i][k]) + "\t"
                    print(string)

        print("*******************************")
        print("============ Y ================")
        print("*******************************")
        Y = []
        for t in range(self.t):
            print("\n\n============ periodo ",t," ============")
            p_list_y=[]
            for p in range(self.p):
                variable = abs(self.Y_p_t[p,t].x)
                p_list_y.append(variable)
                print("produto: ",p," == ", variable)
            Y.append(p_list_y)
        print("\n\n===============================\n\n")

        print("*******************************")
        print("============ X ================")
        print("*******************************")
        X = []
        for t in range(self.t):
            print("\n\n============ periodo ",t," ============")
            p_list_x=[]
            for p in range(self.p):
                p_list_x.append(self.X_p_t[p,t].x)
                print("produto: ",p," == ", self.X_p_t[p,t].x)
            X.append(p_list_x)
        print("\n\n===============================\n\n")

        print("*******************************")
        print("============ I ================")
        print("*******************************")
        I=[]
        for t in range(self.t):
            print("\n\n============ periodo ",t," ============")
            p_list_i=[]
            for i in range(self.i):
                i_list_i=[]
                print("\n============ cliente ",i," ============")
                for p in range(self.p):
                    print("produto: ",p," == ", self.I_p_i_t[p,i,t].x)
                    i_list_i.append(self.I_p_i_t[p,i,t].x)
                p_list_i.append(i_list_i)
            I.append(p_list_i)
        print("\n\n===============================\n\n")

        print("*******************************")
        print("============ R ================")
        print("*******************************")
        R=[]
        for t in range(self.t):
            print("\n\n============ periodo ",t," ============")
            p_list=[]
            for v in range(self.v):
                print("\n============ veiculo ",v," ============")
                v_list=[]
                for i in range(self.i):
                    i_list=[]
                    for k in range(1,self.k):
                        k_list=[]
                        print("\n============ cliente ",i," -> cliente ",k," ============")
                        for p in range(self.p):
                            k_list.append(self.R_p_v_i_k_t[p,v,i,k,t].x)
                            print("produto: ",p," == ", self.R_p_v_i_k_t[p,v,i,k,t].x)
                        i_list.append(k_list)
                    v_list.append(i_list)
                p_list.append(v_list)
            R.append(p_list)
        print("\n\n===============================\n\n")

        print("*******************************")
        print("============ Q ================")
        print("*******************************")
        Q=[]
        for t in range(self.t):
            print("\n\n============ periodo ",t," ============")
            p_list=[]
            for v in range(self.v):
                print("\n============ veiculo ",v," ============")
                v_list=[]
                for i in range(self.i):
                    print("\n============ cliente ",i," ============")
                    i_list=[]
                    for p in range(self.p):
                        print("produto: ",p," == ",self.Q_p_v_i_t[p,v,i,t].x)
                        i_list.append(self.Q_p_v_i_t[p,v,i,t].x)
                    v_list.append(i_list)
                p_list.append(v_list)
            Q.append(p_list)
        print("\n\n===============================\n\n")
    
        return Z,X,Y,I,R,Q,self.model.ObjVal,self.model.MIPGap

    def terminate(self):
        self.model.terminate()

    def solver(self):
        self.createDecisionVariables()
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

        self.model.optimize()