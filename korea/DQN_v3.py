import numpy as np
import pandas as pd
from pathlib import Path
import DQNModel_v3 as dqn

##################### INICIALIZAÇÃO DE VARIÁVEIS ################################################
steps = [] # 9h04 -> 17h50 a cada 5 segundos 
dias = 300 #quantidade de dias que vai rodar
epocas = 100 #quantidade de vezes que vai rodar todos os dias
batch_size = 3 #quantidade de valores para batelada
variaveis = 8 #'preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min', 'IND', 'ISP'
n_entradas = variaveis + 3 #ncont, valor, posicao e inputs
n_saidas = 3 #número de saidas da rede (compra, vende, segura)
custo = 1.06/2 #custo da operação
melhor_reward = 0
caminho_arquivo = "./Consolidado.csv" #caminho para o arquivo de inputs
index_arquivo = ['preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min', 'IND', 'ISP'] #index do arquivo

####################### LEITURA DOS DADOS #######################################################
arquivo = pd.read_csv(caminho_arquivo)
inputs = arquivo[index_arquivo]
pmax = np.amax(inputs.loc[:, inputs.columns[0]]) #define valor minimo do preço
pmin = np.amin(inputs.loc[:, inputs.columns[0]]) #define valor maximo do preço
for i in range(inputs.shape[1]): #roda normalização para todas as colunas
    imax = np.amax(inputs.loc[:, inputs.columns[i]]) #pega valor maximo
    imin = np.amin(inputs.loc[:, inputs.columns[i]]) #pega valor minimo
    inputs.loc[:, inputs.columns[i]] = (inputs.loc[:, inputs.columns[i]] - imin)/(imax - imin) #normaliza preços

dt = arquivo['dt'].values #cria coluna apenas dos dias
steps.append(0)
for i in range(1,len(dt)):
    if (dt[i] != dt[i-1]):
        steps.append(i)     #numero de linhas entre dias
dias = len(steps)

########################  DECLARA MODELO ################################
modelo = dqn.DQNAgent(n_entradas, n_saidas)

########################### FUNÇÕES ###############################################################

def atuacao(preco, ncont, acao, custo, valor):  #preço atual, nº de contratos posicionados,
                                                #ação atual, custo, valor da posição
    valor_cheio = 0.
    ncont_anterior = ncont #salva posição anterior
    ncont += acao #posição atual = pos anterior + ação
    
    if valor != 0:
        valor_cheio = (valor*(pmax-pmin)+pmin)  #valor posicionado atual

    dp = (preco*(pmax-pmin)+pmin) - valor_cheio #variação do preço atual e do preço de compra/venda
    posicao = ncont_anterior*dp*10 - custo*abs(acao) #posicao = lucro - custo (INSTANTÂNEO)

    #calculos sobre o valor    
    if ( (ncont_anterior>=1 and acao==1) or (ncont_anterior<=-1 and acao==-1) ): #aumento do nº de contratos
        valor = abs( (ncont_anterior*valor + acao*preco) / ncont ) #calcula preço medio da posição
    elif ( ncont_anterior==0 and acao != 0 ): #primeiro valor
        valor = preco
    elif  ( ncont==0 ):
        valor = 0
    #caso nao se encaixe nessas condições: valor = valor (nada muda)

    return ncont, valor, posicao, ncont_anterior

def obter_acao(estado):
    decisao = modelo.toma_acao(estado) #calcula a saida da rede neural

    if decisao == 0: #comprar
        if  estado[0][0] < 1: #só compra se não tem nada ainda
            return 1
    elif decisao == 1: #vender
        if  estado[0][0] > -1: #só vende se tiver alguma coisa
            return -1
    return 0 #neutro

def rodar_1dia(precos, custo, dia):
    global melhor_reward
    ncont = 0 #cria variavel de quantidade de contratos
    ncont_anterior = 0 #cria variavel para quantidade de contratos anterior
    valor = 0 #cria variavel para preço medio
    reward = 0 #cria variavel para recompensa
    posicao = 0 #cria variavel de posição 

    for step in range(steps[dia-1], steps[dia]):  #roda os dados
        ultimos_precos = precos[step:step+1] #filtra só o dia que esta
        estado = np.array([np.append([ncont, valor, posicao], ultimos_precos)]) #posição e mercado
        acao = obter_acao(estado) #obtem ação
        ncont, valor, posicao, ncont_anterior = atuacao(precos['preco'][step], ncont, acao, custo, valor)
        if (ncont_anterior != ncont): #reward acumulado recebe reward instantaneo somente se houver lucro/prejuizo real   
            reward += posicao #soma reward
        ultimos_precos = precos[step + 1 : step + 2] #filtra só o dia que esta
        prox_estado = np.array([np.append([ncont, valor, posicao], ultimos_precos)]) #posição e mercado
        modelo.adiciona_memoria(estado, acao, reward, prox_estado, step) #guarda o step na memoria
        if len(modelo.memoria) > batch_size: #se ja tem memoria suficiente
            modelo.treina_modelo(batch_size) #roda o modelo
            
    reward += posicao - custo*abs(ncont) #soma reward - DAY-TRADE (obs: custo nao havia sido considerado no reward pq acao era 0)
    if reward > melhor_reward:
        melhor_reward = reward
    return reward #retorna o valor do reward

def rodar_dias(precos, custo):   
    sum_rewards = 0 #cria variavel de somatoria de recompensas
    for dia in range(1, dias): #loop de dias
        sum_rewards += rodar_1dia(precos, custo, dia) #roda 1 dia e adiciona o total na variavel de somatoria
        print("dia: %0.d: R$ %0.2f" %(dia, sum_rewards)) #mostra o resultado do dia
    return sum_rewards


if __name__ == "__main__":
    try:
        for epoca in range(epocas): #rodar uma quantidade de epocas
            print(f"epoca {epoca}\n") #mostra que epoca vai rodar
            sum_rewards = rodar_dias(inputs, custo) #adiciona o resultado da epoca na somatoria
    finally:
        modelo.salva_pesos('./pesos.h5')
        print(f"Melhor resultado diário: {melhor_reward}")