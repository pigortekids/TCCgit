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
custo = 1.06/2

teste = True   
salvar_pesos = True
directory = str(Path.cwd())
#if teste:
 #   best_pesos = np.load(directory + "/pesos.npy")

####################### LEITURA DOS DADOS #######################################################
arquivo = pd.read_csv(directory + "/Consolidado.csv")
inputs = arquivo['preco'].values
dt = arquivo['dt'].values
imax = np.amax(inputs)
imin = np.amin(inputs)
inputs = (inputs - imin)/(imax - imin) #normaliza preços

steps.append(0)
for i in range(1,len(dt)):
    if (dt[i] != dt[i-1]):
        steps.append(i)     #numero de linhas entre dias
dias = len(steps)
########################### FUNÇÕES ###############################################################

"""environment - implementacao da decisao"""
def atuacao(preco, ncont, acao, custo, valor):  #preço atual, nº de contratos posicionados,
                                         #ação atual, custo, valor da posição
    valor_cheio = 0.
    ncont_anterior = ncont              #salva posição anterior
    ncont += acao                       #posição atual = pos anterior + ação
    
    if valor != 0:
        valor_cheio = (valor*(imax-imin)+imin)
    dp = (preco*(imax-imin)+imin) - valor_cheio            #variação do preço atual e do preço de compra/venda
    
    posicao = ncont_anterior*dp*10 - custo*abs(acao)  #reward = lucro - custo

    #decisoes
    
    if ( (ncont_anterior>=1 and acao==1) or (ncont_anterior<=-1 and acao==-1) ):    #aumento do nº de contratos
        valor = abs( (ncont_anterior*valor + acao*preco) / ncont )     #calcula preço medio da posição
    elif ( ncont_anterior==0 and acao != 0 ):       #primeiro valor
        valor = preco
    elif  ( ncont==0 ):
        valor = 0

    return ncont, valor, posicao, ncont_anterior

"""função para obter a decisão"""
def obter_acao(estado, pesos):
    prod_escalar = np.dot(estado, pesos)
    prod_escalar = np.tanh(prod_escalar)    #tanh do prod_escalar
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
    reward = 0
    ncont_anterior = 0
    posicao = 0
    for step in range(steps[d-1], steps[d]):  #roda os dados
        if step - steps[d-1] > memoria:
            ultimos_precos = precos[step - memoria : step] #filtra só o dia que esta
            estado = np.append([ncont, valor, posicao], ultimos_precos)     #posição e mercado
            acao = obter_acao(estado, pesos)            #obtem ação
            ncont, valor, posicao, ncont_anterior = atuacao(precos[step], ncont, acao, custo, valor)
            if (ncont_anterior != ncont):       #reward acumulado recebe reward instantaneo somente se houver lucro/prejuizo real   
                reward += posicao             #soma reward
    reward += posicao - custo*abs(ncont)            #soma reward - DAY-TRADE (obs: custo nao havia sido considerado no reward pq acao era 0)
    return reward

def rodar_dias(precos, custo):   
    sum_rewards = 0
    #if teste:
     #   pesos = best_pesos
    #else:
    pesos = 2*(np.random.random(memoria + 3) - 0.5)         #gerar novos pesos

    for dia in range(1, dias):                      #loop de dias
        sum_rewards += rodar_1dia(precos, custo, pesos, dia)
        print("dia: %0.d: R$ %0.2f" %(dia,sum_rewards))
    return sum_rewards, pesos

for epoca in range(epocas):
    sum_rewards, pesos = rodar_dias(inputs, custo)
    if sum_rewards > best_rewards:
        best_rewards = sum_rewards
        best_pesos = pesos
    print(f"epoca {epoca}")
    print(f"melhor somatoria de rewards = {best_rewards}\n")
    if teste:
        break
    if salvar_pesos:
        np.save(directory + "/pesos.npy", best_pesos)