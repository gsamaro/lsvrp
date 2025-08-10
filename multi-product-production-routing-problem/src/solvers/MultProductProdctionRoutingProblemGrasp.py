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

class MultProductProdctionRoutingProblemGrasp:

    def __init__(self,map,dir,log:Logger):
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



    def solver():
        return []
    
    def getResults():
        return []

   