import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

INITIAL_ACCOUNT_BALANCE = 10000 #valor inicial na conta
MAX_SHARE_PRICE = 5000 #valor maximo que a ação pode chegar por share
MEMORY = 150 #quantidade de valores anteriores que ele olha
COMPRAR = 0 #define padrão para compra
VENDER = 1 #define padrão para venda

class IgaoEnv(gym.Env): #criação do env
    metadata = {'render.modes': ['human']} #padrão de env da OpenAI

    def __init__(self, df): #função de iniciação da classe
        super(IgaoEnv, self).__init__() #chamada da função de iniciação das classes mães

        self.df = df #pega os valores de entrada
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, MEMORY), dtype=np.float16) #define o formato da entrada
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16) #define o formato da saida 

    def _next_observation(self): #função da proxima observação da IA
        ultimos_valores_dia = self.df.loc[self.current_step - MEMORY + 1 : self.current_step, ['data_sessao','preco_negocio']] #pega ultimos valores
        ultimos_valores_dia = ultimos_valores_dia[ultimos_valores_dia['data_sessao'] == ultimos_valores_dia.loc[self.current_step, 'data_sessao']] #filtra só o dia que esta
        zeros_adicionais = np.zeros(MEMORY - len(ultimos_valores_dia)) #caso tenham dias diferentes preenche de 0
        valores = np.append(zeros_adicionais, ultimos_valores_dia['preco_negocio']) / MAX_SHARE_PRICE
        return valores #retorna essa observação

    def _take_action(self, action): #função para tomar a ação de compra venda ou hold da ação
        current_price = self.df.loc[self.current_step, "preco_negocio"] #pega o valor de agora da ação

        if action == 0: #caso seja uma compra
            shares_bought = COMPRAR #define a variavel de quantidade comprada
            if self.balance - current_price > 0 and self.shares_held == 0: #se tem dinheiro para comprar uma ação
                shares_bought = 1 #define a variavel quantidade para 1 ação
                self.last_balance = self.balance #guarda a informação do balanço
            self.balance -= shares_bought * current_price #atualiza valor da conta
            self.shares_held += shares_bought #atualiza quantidade de shares

        elif action == VENDER: #caso seja uma venda
            shares_sold = 0 #define a variavel de quantidade vendida
            if self.shares_held > 0: #se tem algum share da ação
                shares_sold = 1 #define a variavel de venda como 1 ação
                self.get_reward = True
            self.balance += shares_sold * current_price #atualiza valor da conta
            self.shares_held -= shares_sold #atualiza quantidade de shares

        self.net_worth = self.balance + self.shares_held * current_price #atualiza valor total da carteira

    def step(self, action): #função para cada passo do agente
        self._take_action(action) #toma a ação

        reward = 0
        if self.get_reward:
            reward = self.balance - self.last_balance #define a recompensa do agente
            self.get_reward = False

        self.current_step += 1 #acrescenta em 1 o lugar onde esta no arquivo
        self.steps += 1 #acrescenta em 1 a quantidade de andadas do agente

        done = False #define a variavel de parada do ambiente
        if self.current_step > self.df.shape[0]-1: #se o agente chegou no final do arquivo
            done = True
        elif self.df.loc[self.current_step, "data_sessao"] != self.df.loc[self.current_step-1, "data_sessao"]: #caso mudou o dia acaba
            done = True #define a variavel de parada do ambiente para sim

        obs = self._next_observation() #pega a proxima observação do ambiente

        return obs, reward, done, {} #retorna os valores da função

    def reset(self): #função para resetar o ambiente
        self.balance = INITIAL_ACCOUNT_BALANCE #define o valor da conta como o valor inicial
        self.last_balance = INITIAL_ACCOUNT_BALANCE #define a variavel de ultimo balanço
        self.get_reward = False #define quando ele vai pegar o reward
        self.net_worth = INITIAL_ACCOUNT_BALANCE #define o valor da carteira como o valor inicial
        self.shares_held = 0 #define como 0 a quantidade de shares
        comeco_dia = self.df.drop_duplicates('data_sessao', keep='first').index #pega todos os começos de dia
        self.current_step = comeco_dia[random.randint(0, len(comeco_dia)-1)] #define onde o agente esta no arquivo
        self.steps = 0 #define o valor de passos que o agente realizou

        return self._next_observation() ##retorna a proxima observação do ambiente

    def render(self, mode='human', close=False): #função para mostrar o ambiente
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE #define o valor ganho ate agora
        print(f'Step: {self.steps}') #mostra em que passo esta
        print(f'Balance: {self.balance}') #mostra quanto tem na conta
        print(f'Shares held: {self.shares_held}') #mostra a quantidade de shares
        print(f'Net worth: {self.net_worth}') #mostra o valor da carteira
        print(f'Profit: {profit}') #mostra o valor ganho ate agora
        print("====================================") #mostra uma quebra de linha
