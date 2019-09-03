# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:11:22 2019
@author: Bruno
ALGORITMO DEEP Q-LEARNING NO MERCADO FUTURO
v1
"""
import numpy as np
import pandas as pd
from pathlib import Path
import DQNModel_v1 as dqn

################### REDE NEURAL ################################################
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

class NeuralNetwork:
    def __init__(self, n_entradas, n_saidas, num_neurons):
        self.n_entradas = n_entradas
        self.n_saidas = n_saidas
        self.num_neurons = num_neurons
        self.reset_pesos()
    
    def reset_pesos(self):
        self.weights1 = np.random.rand(self.n_entradas, self.num_neurons) #pesos da camada de entrada para a primeira camada escondida
        self.weights2 = np.random.rand(self.num_neurons, self.n_saidas) #pesos da primeira camada escondida para a saida

    def feedforward(self, inputs):
        self.layer1 = sigmoid(np.dot(inputs, self.weights1)) #valores da entrada para a primeira camada escondida
        self.saida = sigmoid(np.dot(self.layer1, self.weights2)) #valores da primeira camada escondida para a saida
        decisao = np.argmax(self.saida) #pega o maior valor do vetor de saida
        return decisao #retorna a decisão da rede
    
##################### INICIALIZAÇÃO DE VARIÁVEIS ################################################
dias = 0
steps = []   # 9h04 -> 17h50 a cada 5 segundos 
epocas = 100
memoria = 50
variaveis = 6
n_entradas = memoria * variaveis + 3 #ncont, valor, posicao e inputs
n_saidas = 3
n_neuronios = 4
best_rewards = 0
best_pesos = np.zeros(n_entradas)
custo = 1.06/2

teste = True   
salvar_pesos = True
directory = str(Path.cwd())
#if teste:
 #   best_pesos = np.load(directory + "/pesos.npy")

####################### LEITURA DOS DADOS #######################################################
arquivo = pd.read_csv("./Consolidado.csv")
inputs = arquivo[['preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min']]
dt = arquivo['dt'].values
pmax = np.amax(inputs.loc[:, inputs.columns[0]])
pmin = np.amin(inputs.loc[:, inputs.columns[0]])
for i in range(inputs.shape[1]):
    imax = np.amax(inputs.loc[:, inputs.columns[i]])
    imin = np.amin(inputs.loc[:, inputs.columns[i]])
    inputs.loc[:, inputs.columns[i]] = (inputs.loc[:, inputs.columns[i]] - imin)/(imax - imin) #normaliza preços

RNA = NeuralNetwork(n_entradas, n_saidas, n_neuronios) #cria uma rede com os valores do estado como entrada

steps.append(0)
for i in range(1,len(dt)):
    if (dt[i] != dt[i-1]):
        steps.append(i)     #numero de linhas entre dias
dias = len(steps)

########################  DECLARA MODELO ################################
modelo = dqn.DQNAgent(n_saidas, n_neuronios)

########################### FUNÇÕES ###############################################################

def atuacao(preco, ncont, acao, custo, valor):  #preço atual, nº de contratos posicionados,
                                         #ação atual, custo, valor da posição
    valor_cheio = 0.
    ncont_anterior = ncont              #salva posição anterior
    ncont += acao                       #posição atual = pos anterior + ação
    
    if valor != 0:
        valor_cheio = (valor*(pmax-pmin)+pmin)  #valor posicionado atual

    dp = (preco*(pmax-pmin)+pmin) - valor_cheio            #variação do preço atual e do preço de compra/venda
    posicao = ncont_anterior*dp*10 - custo*abs(acao)       #posicao = lucro - custo (INSTANTÂNEO)

    #calculos sobre o valor    
    if ( (ncont_anterior>=1 and acao==1) or (ncont_anterior<=-1 and acao==-1) ):    #aumento do nº de contratos
        valor = abs( (ncont_anterior*valor + acao*preco) / ncont )     #calcula preço medio da posição
    elif ( ncont_anterior==0 and acao != 0 ):       #primeiro valor
        valor = preco
    elif  ( ncont==0 ):
        valor = 0
    #caso nao se encaixe nessas condições: valor = valor (nada muda)

    return ncont, valor, posicao, ncont_anterior

"""função para obter a decisão"""
def obter_acao(estado):
    decisao = RNA.feedforward(estado) #calcula a saida da rede neural

    if decisao == 0: #comprar
        if  estado[0] < 1:
            return 1
    elif decisao == 1: #vender
        if  estado[0] > -1:
            return -1
    return 0 #neutro

def rodar_1dia(precos, custo, dia):
    ncont = 0       # reinicia nº de contratos
    valor = 0       # reinicia preço medio
    reward = 0
    posicao = 0
    ncont_anterior = 0
    for step in range(steps[dia-1], steps[dia]):  #roda os dados
        if step - steps[dia-1] > memoria:
            ultimos_precos = precos[step - memoria : step] #filtra só o dia que esta
            estado = np.append([ncont, valor, posicao], ultimos_precos)     #posição e mercado
            acao = modelo(estado)            #obtem ação
            ncont, valor, posicao, ncont_anterior = atuacao(precos['preco'][step], ncont, acao, custo, valor)
            if (ncont_anterior != ncont):       #reward acumulado recebe reward instantaneo somente se houver lucro/prejuizo real   
                reward += posicao             #soma reward           
            
    reward += posicao - custo*abs(ncont)            #soma reward - DAY-TRADE (obs: custo nao havia sido considerado no reward pq acao era 0)
    return reward

def rodar_dias(precos, custo):   
    sum_rewards = 0

    for dia in range(1, dias):                              #loop de dias
        sum_rewards += rodar_1dia(precos, custo, dia)
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