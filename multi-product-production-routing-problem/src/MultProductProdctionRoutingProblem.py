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
            for k in range(self.k):
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
        
    def outModel(self):
        self.model.Params.OutputFlag = 1
        self.model.write("./out/modelo.lp")

    def solver(self):
        self.createDecisionVariables()
        self.crateObjectiveFunction()
        self.outModel()