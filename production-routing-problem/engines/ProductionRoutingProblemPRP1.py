#################################################################################################
# ProductionRoutingProblemPRP1
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
from engines.dtos.Data import Data

class ProductionRoutingProblemPRP1:
    def __init__(self, data = Data):
        self.model = gp.Model("Production_Routing_Problem")
        self.CONST_PERIOD_T=data.CONST_PERIOD_T
        self.CONST_CUSTOMER_I=data.CONST_CUSTOMER_I
        self.CONST_CUSTOMER_J=data.CONST_CUSTOMER_J
        self.data = data
        self.p = {}
        self.I = {}
        self.y = {}
        self.z = {}
        self.x = {}
        self.q = {}
        self.w = {}
        self.M = []
        self.MT = []

    def createDecisionVariables(self):
        for t in range(self.CONST_PERIOD_T):
            self.p[t] = self.model.addVar(vtype=GRB.INTEGER, name=f"p[{t}]")
            self.y[t] = self.model.addVar(vtype=GRB.BINARY, name=f"y[{t}]") 
           
        for i in range(self.CONST_CUSTOMER_I):
            for j in range(self.CONST_CUSTOMER_J):
                for t in range(self.CONST_PERIOD_T):
                    self.x[i,j,t] = self.model.addVar(vtype=GRB.BINARY, name=f"x[{i},{j},{t}]") 
        
        for i in range(self.CONST_CUSTOMER_I):
            for t in range(self.CONST_PERIOD_T):
                self.I[i,t] = self.model.addVar(vtype=GRB.INTEGER, name=f"I[{i},{t}]") 
                self.w[i,t] = self.model.addVar(vtype=GRB.INTEGER, name=f"w[{i},{t}]") 
                self.q[i,t] = self.model.addVar(vtype=GRB.INTEGER, name=f"q[{i},{t}]")

                if(i==0):
                    self.z[0,t] = self.model.addVar(vtype=GRB.INTEGER, name=f"z[{i},{t}]") 
                else:
                    self.z[i,t] = self.model.addVar(vtype=GRB.BINARY, name=f"z[{i},{t}]") 
    
    def crateObjectiveFunction(self):
        objExpr = gp.LinExpr()
        
        for t in range(self.CONST_PERIOD_T):
            objExpr_1 = gp.LinExpr()
            for i in range(self.CONST_CUSTOMER_I):
                objExpr_1+= self.data.h[i] + self.I[i,t]

            objExpr_2 = gp.LinExpr()
            for i in range(self.CONST_CUSTOMER_I):
                for j in range(self.CONST_CUSTOMER_J):
                   objExpr_2+= self.data.c[i][j]*self.x[i,j,t] 
                   
            objExpr += self.data.u * self.p[t] + self.data.f * self.y[t] + objExpr_1 + objExpr_2

        self.model.setObjective(objExpr, GRB.MINIMIZE)


    def createRestrictionBalanceStockFlowFactoryAndCustomers(self):
        for i in range(self.CONST_CUSTOMER_I):
            self.model.addConstr(self.I[i, 0] == self.data.I0, name=f"rest_1_{i}")
        
        for t in range(1,self.CONST_PERIOD_T):
            r2 = 0
            for i in range(self.CONST_CUSTOMER_I):
                r2+=self.q[i,t] + self.I[0,t]
                self.model.addConstr(  self.I[i, t-1] + self.q[i,t] == self.data.d[i][t]+self.I[i,t], name=f"rest_3_{t}_{i}")
            self.model.addConstr(  self.I[0, t-1] + self.p[t] == r2, name=f"rest_2_{t}_{i}")

    def createRestrictionProductionCapactiy(self):
        for t in range(self.CONST_PERIOD_T): 
            self.model.addConstr( self.p[t] <= self.M[t]*self.y[t], name=f"rest_4_{t}")

    def createRestrictionLimitMaximumInventory(self):
        for t in range(self.CONST_PERIOD_T): 
            self.model.addConstr( self.I[0,t] <= self.data.L[0], name=f"rest_5_{t}")
        
        for t in range(1,self.CONST_PERIOD_T): 
            for i in range(self.CONST_CUSTOMER_I):
                self.model.addConstr( self.I[i,t-1] + self.q[i,t] <= self.data.L[i], name=f"rest_6_{t}")

    def createRestrictionAllowPositiveDeliveryQuantityOnly(self):
        for t in range(1,self.CONST_PERIOD_T): 
            for i in range(self.CONST_CUSTOMER_I):
                self.model.addConstr( self.q[i,t]<=self.MT[i][t]*self.z[i,t], name=f"rest_7_{t}_{i}")

    def createRestrictionEachCustomerCanBeVistitedByMostOneVehicle(self):
        for t in range(self.CONST_PERIOD_T):
            for i in range(self.CONST_CUSTOMER_I):
                r8=0
                for j in range(self.CONST_CUSTOMER_J):
                    r8+=self.x[i,j,t]
                self.model.addConstr(r8==self.z[i,t], name=f"rest_8_{t}_{i}")

    def createRestrictionVehicleFlowConservation(self):
        for t in range(self.CONST_PERIOD_T):
            for i in range(self.CONST_CUSTOMER_I):
                r9_1=0
                for j in range(self.CONST_CUSTOMER_J):
                    r8+=self.x[i,j,t]
        
                r9_2=0
                for j in range(self.CONST_CUSTOMER_J):
                    r8+=self.x[j,i,t]

                self.model.addConstr(r9_1+r9_2 == 2*self.z[i,t], name=f"rest_9_{t}_{i}")

    def createRestrictionLimitNumberTrucks(self):
        for t in range(self.CONST_PERIOD_T):
            self.model.addConstr(self.z[0,t]<self.data.m, name=f"rest_10_{t}")

    def createRestrictionVehicleLoadingAndSubTourElimination(self):  
        for i in range(self.CONST_CUSTOMER_I):
            for j in range(self.CONST_CUSTOMER_J):
                for t in range(self.CONST_PERIOD_T):
                    self.model.addConstr(self.w[i,t] - self.w[j,t] >= self.q[i,t] - self.MT[i][j]*(1 - self.x[i,j,t]), name=f"rest_11_{t}")

        for t in range(self.CONST_PERIOD_T):
            for i in range(self.CONST_CUSTOMER_I):
                self.model.addConstr(self.w[i,t] <= self.data.Q*self.z[i,t], name=f"rest_11_{t}")


    def solve(self):
        self.createDecisionVariables()
        self.crateObjectiveFunction()
        self.createRestrictionBalanceStockFlowFactoryAndCustomers()
        self.createRestrictionProductionCapactiy()
        self.createRestrictionLimitMaximumInventory()
        self.createRestrictionAllowPositiveDeliveryQuantityOnly()
        self.createRestrictionEachCustomerCanBeVistitedByMostOneVehicle()
        self.createRestrictionVehicleFlowConservation()
        self.createRestrictionLimitNumberTrucks()
        self.createRestrictionVehicleLoadingAndSubTourElimination()