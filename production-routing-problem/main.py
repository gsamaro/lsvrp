import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.boudia_et_al.ReadingFunction import ReadingFunction
from tests.ProductionRoutingProblemPRP1Tests import writeDto
from engines.ProductionRoutingProblemPRP1 import ProductionRoutingProblemPRP1 as PRP1
from interface.Drawer import Drawer


def __main__():

    drawer = Drawer()
    drawer.mainloop()
    info = drawer.getInformations()
    
    read = ReadingFunction(info['clients'],info['instance'])
    data = read.oultsp()
    prp = PRP1(data).solve()


    if(info['validete']=="SIM"):
        writeDto(data,info['clients'],info['instance'])
        data.toString()

    return 0

__main__()