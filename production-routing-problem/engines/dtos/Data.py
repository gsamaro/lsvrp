class Data:
    def __init__(self,costPeriodT, costCustomerI, costCustomerJ):
        self.u = 0
        self.f = 0
        self.h = []
        self.c = [[]]
        self.d = [[]]
        self.C = 0
        self.Q = 0
        self.L = []
        self.I0 = 0
        self.m = 0
        self.CONST_PERIOD_T=costPeriodT
        self.CONST_CUSTOMER_I=costCustomerI
        self.CONST_CUSTOMER_J=costCustomerJ

    def setUnitProductionCost(self,u):
        self.u = u
    def setFixedProductionCost(self,f):
        self.f = f
    def setUnitInventoryHoldingCost(self,h):
        self.h = h
    def setTransportationCost(self,c):
        self.c = c
    def setDemandCustomer(self,d):
        self.d = d
    def setProductionCapacity(self,C):
        self.C = C
    def setVehicleCapactiy(self,Q):
        self.Q = Q
    def setMaximumTargetInvetoryLevel(self,L):
        self.L = L
    def setInitalInventory(self,I0):
        self.I0 = I0
    def setMaxTrucks(self,m):
        self.m = m

    def toString(self):
        print("Production Capacity :",self.C)
        print("Vehicle Capactiy :",self.Q)
        print("Production Cost: ", self.u)
        print("Fixed Production Cost: ", self.f)
        print("Inital Inventory :",self.I0)
        print("Unit Inventory Holding Cost :", {'size':len(self.h),'value':self.h})
        print("Maximum Target InvetoryLevel :", {'size':len(self.L),'value':self.L})
        print("Demand Customer size:",len(self.d))
        for d in self.d:
            print("\t",{'size':len(d)})
        print("Transportation Cost size :",len(self.c))
        for c in self.c:
            print("\t",{'size':len(c)})
        