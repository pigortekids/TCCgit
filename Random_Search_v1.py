# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:11:22 2019

@author: Bruno

ALGORITMO RANDOM SEARCH NO MERCADO FUTURO
v1
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn import preprocessing

##################### INICIALIZAÇÃO DE VARIÁVEIS ################################################
dias = 272
steps = 6431   # 9h04 -> 17h50 a cada 5 segundos 
epocas = 10000
janela = 2      # numero entradas na janela de tempo
best_reward = 0
best_W = [0.0,0.0,0.0]

####################### LEITURA DOS DADOS #######################################################
arquivo = pd.read_csv("D:/TCC/ArquivoFinal/v1/Consolidado.csv")
inputs = arquivo['preco'].values
imax = np.amax(inputs)
inputs = inputs/imax

########################### FUNÇÕES ###############################################################

"""environment - implementacao da decisao"""
def atuacao(pt, ncont, acao, c, valor):  #preço atual, nº de contratos posicionados,
                                         #ação atual, custo, valor da posição
    ncont_anterior = ncont              #salva posição anterior
    ncont += acao                       #posição atual = pos anterior + ação
    dp = (pt - abs(valor))*imax            #variação do preço atual e do preço de compra/venda
    recompensa = ncont_anterior*dp*10 - c*abs(acao)  #reward = lucro - custo
    if ((ncont > 0 and acao == 1) or (ncont < 0 and acao == -1)):    
        valor = abs( (ncont_anterior*valor + acao*pt) / ncont )     #calcula preço medio da posição
    elif ncont == 0:
        valor = 0                  #se não ha posição
        
    return ncont, valor, recompensa

"""função para obter a decisão"""
def obter_acao(estado, W):
    prod_escalar = np.dot(estado, W)
    
    if prod_escalar > 0.5:
        if  estado[0] < 1:
            return 1            #comprar
        else:
            return 0            #limita a posição entre -1 e 1
    elif prod_escalar < -0.5:
        if  estado[0] > -1:
            return -1           #vender
        else:
            return 0            #limita a posição entre -1 e 1
    else:
        return 0                #neutro

def rodar_1dia(p, c, W):
    ncont = 0       #reinicia nº de contratos
    valor = 0       # reinicia preço medio
    r = 0
    
    global best_W, best_reward
    
    for step in range(steps-1):                 #roda os dados
        estado = ([ncont, valor, p[step]])      #posição e mercado
        acao = obter_acao(estado, W)            #obtem ação
        ncont, valor, reward = atuacao(p[step], ncont, acao, c, valor)
        if reward > best_reward:
            best_W = W
            best_reward = reward
        r += reward             #soma reward
    return r

def rodar_dias(p, c):
    r = []
    for item in range(10):              #loop de dias
        W = np.random.random(3)         #gerar novos pesos
        r.append(rodar_1dia(p, c, W))   #adiciona na lista de rewards
    return r


print(rodar_dias(inputs, 1.06))
print(best_reward)
print(best_W)