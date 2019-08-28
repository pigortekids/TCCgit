# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:11:22 2019

@author: Bruno

ALGORITMO RANDOM SEARCH NO MERCADO FUTURO
v1
"""
import numpy as np
import pandas as pd
from pathlib import Path

##################### INICIALIZAÇÃO DE VARIÁVEIS ################################################
dias = 0
steps = []   # 9h04 -> 17h50 a cada 5 segundos 
epocas = 10000
memoria = 50
best_rewards = 0
best_pesos = np.zeros(memoria + 2)

teste = True   
salvar_pesos = True
directory = str(Path.cwd())
if teste:
    best_pesos = np.load(directory + "/pesos.npy")

####################### LEITURA DOS DADOS #######################################################
arquivo = pd.read_csv("C:/Users/igora/Desktop/Consolidado.csv")
inputs = arquivo['preco'].values
dt = arquivo['dt'].values
imax = np.amax(inputs)
inputs = inputs/imax        #normaliza preços

steps.append(0)
for i in range(1,len(dt)):
    if (dt[i] != dt[i-1]):
        steps.append(i)     #numero de linhas entre dias
dias = len(steps)
########################### FUNÇÕES ###############################################################

"""environment - implementacao da decisao"""
def atuacao(preco, ncont, acao, custo, valor):  #preço atual, nº de contratos posicionados,
                                         #ação atual, custo, valor da posição
    ncont_anterior = ncont              #salva posição anterior
    ncont += acao                       #posição atual = pos anterior + ação
    dp = (preco - abs(valor))*imax            #variação do preço atual e do preço de compra/venda
    recompensa = ncont_anterior*dp*10 - custo*abs(acao)  #reward = lucro - custo
    if ((ncont > 0 and acao == 1) or (ncont < 0 and acao == -1)):    
        valor = abs( (ncont_anterior*valor + acao*preco) / ncont )     #calcula preço medio da posição
    elif ncont == 0:
        valor = 0                  #se não ha posição
        
    return ncont, valor, recompensa

"""função para obter a decisão"""
def obter_acao(estado, pesos):
    prod_escalar = np.dot(estado, pesos)
    
    if prod_escalar > 0.5:
        if  estado[0] < 1:
            return 1            #comprar
    elif prod_escalar < -0.5:
        if  estado[0] > -1:
            return -1           #vender
    return 0                #neutro

def rodar_1dia(precos, custo, pesos, d):
    ncont = 0       # reinicia nº de contratos
    valor = 0       # reinicia preço medio
    sum_reward = 0
    for step in range(steps[d-1], steps[d]):  #roda os dados
        if step - steps[d-1] > memoria:
            ultimos_precos = precos[step - memoria : step] #filtra só o dia que esta
            estado = np.append([ncont, valor], ultimos_precos)     #posição e mercado
            acao = obter_acao(estado, pesos)            #obtem ação
            ncont, valor, reward = atuacao(precos[step], ncont, acao, custo, valor)
            sum_reward += reward             #soma reward
    return sum_reward

def rodar_dias(precos, custo):
    pesos = np.random.random(memoria + 2)         #gerar novos pesos
    if teste:
        pesos = best_pesos
    for dia in range(1, dias):              #loop de dias
        sum_rewards = rodar_1dia(precos, custo, pesos, dia)
    return sum_rewards, pesos

for epoca in range(epocas):
    sum_rewards, pesos = rodar_dias(inputs, 1.06)
    if sum_rewards > best_rewards:
        best_rewards = sum_rewards
        best_pesos = pesos
    print(f"epoca {epoca}")
    print(f"melhor somatoria de rewards = {best_rewards}\n")
    if teste:
        break
    if salvar_pesos:
        np.save(directory + "/pesos.npy", best_pesos)