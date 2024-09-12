from src.ReadPrpFile import ReadPrpFile as RD
from src.MultProductProdctionRoutingProblem import MultProductProdctionRoutingProblem as MPPRP


def __main__():
    # Exemplo de uso
    file_path = './data/DATA_PRP_5C/PRP1_C5_P3_V1_T6_S1.dat'
    read = RD(file_path)
    mpprp = MPPRP(read.getDataSet()).solver()


__main__()