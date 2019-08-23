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

##################### INICIALIZAÇÃO DE VARIÁVEIS ################################################
dias = 286
steps = 6431   # 9h04 -> 17h50 a cada 5 segundos 
epocas = 10000
janela = 2      # numero entradas na janela de tempo

########################### FUNÇÕES ###############################################################

"""environment - implementacao da decisao"""
def atuacao(pt, ncont, acao, c, valor):  #preço atual, nº de contratos posicionados,
                                         #ação atual, custo, valor da posição
    ncont_anterior = ncont              #salva posição anterior
    ncont += acao                       #posição atual = pos anterior + ação
    dp = pt - abs(valor)                #variação do preço atual e do preço de compra/venda
    recompensa = ncont_anterior*dp - c*abs(acao)  #reward = lucro - custo
    if ncont != 0:    
        valor = abs( (ncont_anterior*valor + acao*pt) / ncont )     #calcula preço medio da posição
    else:
        valor = 0           #se não ha posição
        
    return ncont, valor, recompensa

def obter_acao(estado, W):
    
    prod_escalar = estado.dot(W)
    
    if prod_escalar > 0.5:
        return 1
    elif prod_escalar < -0.5:
        return -1
    else:
        return 0
        
def rodar_1dia():
    ncont = 0

print("n = %0.0f contratos" %n)
print("preço médio = %0.1f" %v)
print("reward = %0.3f" %r)