from re import I
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import copy
import numpy as np
import json

def gera_candidatos(x_c,y_c,lado):
  return [(x_c+lado, y_c),(x_c,y_c+lado),(x_c+lado,y_c+lado),(x_c-lado,y_c-lado)]

def calcular_distancia(ponto1, ponto2):
    x1, y1 = ponto1
    x2, y2 = ponto2
    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distancia

def calcula_sobreposicao(quadrado1, quadrado2):
    x1, y1, lado1 = quadrado1
    x2, y2, lado2 = quadrado2
    colisao_x = (x1 < x2 + lado2) and (x1 + lado1 > x2)
    colisao_y = (y1 < y2 + lado2) and (y1 + lado1 > y2)
    return colisao_x and colisao_y

def sobreposto(squares, candidato, l_c):
  for square in squares:
    x, y, lado = square
    if(calcula_sobreposicao((x,y,lado),(candidato[0],candidato[1], l_c))):
        return True

  return False
def escolhe_candidato(candidatos,  squares, l_c):

  if(len(candidatos)==0):
    return (0,0)
 
  v = float('inf')
  ind = 0
  for i in range(len(candidatos)):

    # xr, yr = ponto_central(squares)
    xr = 0
    yr = 0

    c = calcular_distancia( (xr, yr), candidatos[i])

    if(c < v ):
      if(sobreposto(squares, candidatos[i], l_c)==False):
        v = c
        ind = i
  
  escolha  = candidatos[ind]
  candidatos.pop(ind)
  return escolha

def gere_squares(radii = []):
  radii = sorted(radii, reverse=True)
  # random.shuffle(radii)
  x_c = 0
  y_c = 0
  squares=[]
  candidatos = []
  for i in range(len(radii)):
    l_c = radii[i]*2
 
    escolhido = escolhe_candidato(candidatos, squares, l_c)
  
    x_c = escolhido[0]
    y_c = escolhido[1]
    candidatos.extend(gera_candidatos(x_c, y_c,l_c))

    squares.append([x_c,y_c,l_c])
    
  return squares, candidatos

def ponto_central(squares):
    if(len(squares)==0):
      return 0,0
    centros = []
    for square in squares:
      x, y, lado = square
      centros.append((x+(lado/2),y+(lado/2)))
    # Calcula a média das coordenadas x e y dos centros
    x_media = np.mean([centro[0] for centro in centros])
    y_media = np.mean([centro[1] for centro in centros])
    return x_media, y_media

def ponto_central_entre_circulos(squares):
    
    if(len(squares)==0):
      return 0,0

    centros = []
    raios = []
    for square in squares:
      x, y, lado = square
      centros.append((x+(lado/2),y+(lado/2)))
      raios.append(lado/2)
    # Pondera as coordenadas x e y pelos quadrados dos raios
    x_ponderado = np.sum([centro[0] * raio**2 for centro, raio in zip(centros, raios)])
    y_ponderado = np.sum([centro[1] * raio**2 for centro, raio in zip(centros, raios)])
    
    # Calcula a soma dos quadrados dos raios
    soma_raios_quadrados = np.sum([raio**2 for raio in raios])

    # Calcula o ponto central ponderado
    x_media = x_ponderado / soma_raios_quadrados
    y_media = y_ponderado / soma_raios_quadrados

    return x_media, y_media

def ajustarR(r, squares):
  xr,yr,R = r

  ponto_central = ponto_central_entre_circulos(squares)
  xr = ponto_central[0]
  yr = ponto_central[1]
  rr = 0
  stop = False
  while(stop == False):
    rr=rr+0.1

    todos = []
    for square in squares:
      x, y, lado = square
      xi = x+(lado/2)
      yi = y+(lado/2)
      ri = lado/2
      if((xr - xi)**2 + (yr - yi)**2 <= (rr - ri)**2):
        todos.append(True)
      else:
        todos.append(False)
         
    if(all(todos)):
      stop = True

  return [xr,yr, rr]
    
def calculaR(squares):
    lado_R = 0
    for square in squares:
      x, y, lado = square
      if(lado_R<x+lado):
        lado_R = x+lado
      elif(lado_R<y+lado):
        lado_R = y+lado

    R = lado_R * math.sqrt(2) / 2
    xr = lado_R/2
    yr = lado_R/2
    return [xr,yr, R]

def plot_squares(squares, R, name):
    fig, ax = plt.subplots()
    xr,yr,rr = R
    points = []
    for square in squares:
        x, y, lado = square
        # square_patch = patches.Rectangle((x, y), lado, lado, fill=False, color='blue', linewidth=2)
        # ax.add_patch(square_patch)
        circle = plt.Circle([(x+(lado/2)),(y+(lado/2))], lado/2, fill=False)
        ax.add_patch(circle)
        point = {
            "x":x+(lado/2),
            "y":x+(lado/2),
            "r":x+(lado/2)
        }
        points.append(point)
    # square_patch = patches.Rectangle((0, 0), R[0], R[1], fill=False, color='red', linewidth=2)
    # ax.add_patch(square_patch)
    # Criar um círculo
    circle = patches.Circle((R[0], R[1]), R[2], fill=False)
    ax.add_patch(circle)
    ax.set_xlim([ xr-rr, xr+rr ])
    ax.set_ylim([ yr-rr, yr+rr])
    plt.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    
    awnser = {
        "points": points,
        "R": rr,
        "R_x": xr,
        "R_y": yr,
        "fo": rr
    }
    # Exibir o gráfico
    plt.title('Plotagem de Círculos')
    plt.savefig("./"+name+'-mosaico.png')
   
    lp = "./"+name+"-mosaico.json"
    with open(lp, 'a') as arquivo:
        arquivo.write(json.dumps(awnser)) 

if __name__ == "__main__":
    radii = [1,1]
    squares, candidatos = gere_squares(radii)
    rr = calculaR(squares)
    rr = ajustarR(rr, squares)
    plot_squares(squares, rr,"test")

    radii = [1,2,3,4,5,6,7]
    squares, candidatos = gere_squares(radii)
    rr = calculaR(squares)
    rr = ajustarR(rr, squares)
    plot_squares(squares, rr,"instancia_1")

    radii = [1]*5 + [2]*5 + [3]*5
    squares, candidatos = gere_squares(radii)
    rr = calculaR(squares)
    rr = ajustarR(rr, squares)
    plot_squares(squares, rr,"instancia_2")

    radii = [1]*10 + [2]*10 + [3]*10
    squares, candidatos = gere_squares(radii)
    rr = calculaR(squares)
    rr = ajustarR(rr, squares)
    plot_squares(squares, rr,"instancia_3")

    radii = [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10
    squares, candidatos = gere_squares(radii)
    rr = calculaR(squares)
    rr = ajustarR(rr, squares)
    plot_squares(squares, rr,"instancia_4")

    radii = [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10 + [6]*10 + [7]*10 + [8]*10 + [9]*10 + [10]*10
    squares, candidatos = gere_squares(radii)
    rr = calculaR(squares)
    rr = ajustarR(rr, squares)
    plot_squares(squares, rr,"instancia_5")
    



    
       





    



