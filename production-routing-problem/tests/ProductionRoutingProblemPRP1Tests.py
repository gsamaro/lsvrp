import json

from engines.dtos.Data import Data


def writeDto(data=Data, c="", i=""):
    with open("./out/dtoValidate/" + c + "_clients_" + i + ".json", "w") as arquivo:
        arquivo.write(json.dumps(data.__dict__))
