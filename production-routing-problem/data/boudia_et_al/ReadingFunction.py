import math
from engines.dtos.Data import Data

class ReadingFunction:
    def __init__(self,clients="",instances=""):
        self.clients = clients
        self.instances = instances

    def read(self):
        dto=object
        instances = "data/boudia_et_al/"+self.clients+"_Clients/"+self.instances+"/instances.txt"
        km=0
        points=[]
        
        with open(instances, 'r',encoding='latin-1') as file:
            lines = file.readlines()
            dto = Data(int(lines[2].split(':')[1].strip()),int(lines[0].split(':')[1].strip())+1,int(lines[0].split(':')[1].strip())+1)
            points.append((0,0))
            demandT=[[0]*dto.CONST_PERIOD_T]
            for line in lines[6:]:
                info = line.split()
                demand = []
                for d in range(4,len(info)):
                    demand.append(int(info[d]))
                demandT.append(demand)
                points.append((float(info[2]),float(info[3])))
            dto.setDemandCustomer(demandT)
            km = int(lines[4].split(':')[1].strip())
        parameters = "data/boudia_et_al/"+self.clients+"_Clients/"+self.instances+"/parameters.txt"
        with open(parameters, 'r',encoding='latin-1') as file:
            lines = file.readlines()
            dto.setMaxTrucks(int(lines[7].split(':')[1].strip()))
            dto.setVehicleCapactiy(int(lines[6].split(':')[1].strip()))
            dto.setProductionCapacity(int(lines[0].split(':')[1].strip()))
            dto.setInitalInventory(0)
            dto.setUnitProductionCost(1)
            dto.setUnitInventoryHoldingCost( [int(lines[4].split(':')[1].strip())]*dto.CONST_CUSTOMER_I)
            dto.setFixedProductionCost(int((lines[3].split(':')[1].strip())))
            store = [0]
            for line in lines[10:]:
                info=line.split(' ')
                for i in range(int(info[1])-1,int(info[3])):
                    store.append(int(info[8]))
            dto.setMaximumTargetInvetoryLevel(store)
        dto.setTransportationCost(self.getDirections(points,km))
        return dto
    
    def getDistence(self,origin,destination):
        x1, y1 = origin
        x2, y2 = destination
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def getDirections(self,points,km):
        directions = []
        for origin in points:
            line = []
            for destination in points:
                line.append((self.getDistence(origin,destination)*km))
            directions.append(line)    
        return directions
    
    def oultsp(self):
        return self.read()