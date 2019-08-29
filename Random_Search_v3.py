# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 23:11:22 2019

@author: Bruno

ALGORITMO RANDOM SEARCH NO MERCADO FUTURO
v3
"""
import numpy as np
import pandas as pd
from pathlib import Path

################### REDE NEURAL ####################
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

class NeuralNetwork:
    def __init__(self, n_entradas, n_saidas, num_neurons, random = True):
        self.n_entradas = n_entradas
        self.n_saidas = n_saidas
        self.num_neurons = num_neurons
        self.random = random
        self.reset_pesos()
    
    def reset_pesos(self):
        self.weights1 = np.random.rand(self.n_entradas, self.num_neurons) #pesos da camada de entrada para a primeira camada escondida
        self.weights2 = np.random.rand(self.num_neurons, self.n_saidas) #pesos da primeira camada escondida para a saida

    def feedforward(self, inputs):
        if self.random:
            self.reset_pesos()
        self.layer1 = sigmoid(np.dot(inputs, self.weights1)) #valores da entrada para a primeira camada escondida
        self.saida = sigmoid(np.dot(self.layer1, self.weights2)) #valores da primeira camada escondida para a saida
        print(self.saida)
        decisao = np.argmax(self.saida) #pega o maior valor do vetor de saida
        print(decisao)
        return decisao #retorna a decisão da rede
    
##################### INICIALIZAÇÃO DE VARIÁVEIS ################################################
dias = 0
steps = []   # 9h04 -> 17h50 a cada 5 segundos 
epocas = 100
memoria = 50
n_entradas = memoria + 2
n_saidas = 3
n_neuronios = 4
best_rewards = 0
best_pesos = np.zeros(n_entradas)

teste = True   
salvar_pesos = True
directory = str(Path.cwd())
#if teste:
 #   best_pesos = np.load(directory + "/pesos.npy")

####################### LEITURA DOS DADOS #######################################################
arquivo = pd.read_csv("./Consolidado.csv")
inputs = arquivo['preco'].values
dt = arquivo['dt'].values
imax = np.amax(inputs)
imin = np.amin(inputs)
inputs = (inputs - imin)/(imax - imin) #normaliza preços
custo = 1.06 / 2 #custo
RNA = NeuralNetwork(n_entradas, n_saidas, n_neuronios) #cria uma rede com os valores do estado como entrada

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
    
    return ncont, valor, recompensa, ncont_anterior

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
    sum_reward = 0
    reward = 0
    ncont_anterior = 0
    for step in range(steps[dia-1], steps[dia]):  #roda os dados
        if step - steps[dia-1] > memoria:
            ultimos_precos = precos[step - memoria : step] #filtra só o dia que esta
            estado = np.append([ncont, valor], ultimos_precos)     #posição e mercado
            acao = obter_acao(estado)            #obtem ação
            ncont, valor, reward, ncont_anterior = atuacao(precos[step], ncont, acao, custo, valor)
            if (ncont_anterior != ncont):       #reward acumulado recebe reward instantaneo somente se houver lucro/prejuizo real   
                sum_reward += reward             #soma reward                           
                         
    sum_reward += reward - custo*abs(ncont)            #soma reward - DAY-TRADE (obs: custo nao havia sido considerado no reward pq acao era 0)
    return sum_reward

def rodar_dias(precos, custo):   
    sum_rewards = 0

    for dia in range(1, dias):                      #loop de dias
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