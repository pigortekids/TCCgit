import numpy as np
import pandas as pd
import DQNModel_v4 as dqn
import matplotlib.pyplot as plt

##################### INICIALIZAO DE VARIAVEIS ################################################
steps = [] # 9h04 -> 17h50 a cada 5 segundos 
epocas = 10000 #quantidade de vezes que vai rodar todos os dias
janela = 10 #janela de valores
n_variaveis = 8 #'preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min', 'IND', 'ISP'
n_entradas = n_variaveis * janela + 3 #ncont, valor, posicao e inputs
n_neuronios = 64 #numero de neuronios da camada escondida
n_saidas = 3 #nmero de saidas da rede (compra, vende, segura)
custo = 1.06/2 #custo da operao
melhor_reward = 0
caminho_arquivo = "C:/Users/Odete/Desktop/consolidado.csv"
#caminho_arquivo = "./consolidado2.csv" #caminho para o arquivo de inputs
index_arquivo = ['preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min', 'IND', 'ISP'] #index do arquivo
epsilon = 1.0 #valor de epsilon
epsilon_min = 0.0001 #valor minimo de epsilon
epsilon_decay = (epsilon - epsilon_min) / epocas #o valor que vai retirado do epsilon por epoca
rewards = [0] #variavel para guardar rewards
plotx = [0] #variavel para guardar valores a serem plotados do eixo x
teste = True
qnt_testes = 10

####################### LEITURA DOS DADOS #######################################################
arquivo = pd.read_csv(caminho_arquivo)
inputs = arquivo[index_arquivo]
pmax = np.amax( inputs.loc[:, inputs.columns[0]] ) #define valor minimo do preo
pmin = np.amin( inputs.loc[:, inputs.columns[0]] ) #define valor maximo do preo

for i in range( inputs.shape[1] ): #roda normalizo para todas as colunas
    imax = np.amax( inputs.loc[:, inputs.columns[i]] ) #pega valor maximo
    imin = np.amin( inputs.loc[:, inputs.columns[i]] ) #pega valor minimo
    inputs.loc[:, inputs.columns[i]] = ( inputs.loc[:, inputs.columns[i]] - imin ) / ( imax - imin ) #normaliza prs

dt = arquivo['dt'].values #cria coluna apenas dos dias
steps.append(0)
for i in range( 1, len(dt) ):
    if (dt[i] != dt[i-1]):
        steps.append(i)     #numero de linhas entre dias
dias = len(steps)

########################  DECLARA MODELO ################################
modelo = dqn.DQNAgent(n_entradas, n_saidas, epsilon, janela, n_neuronios, n_variaveis)

########################### FUNCOES ###############################################################

def atuacao( preco, ncont, acao, custo, valor ):  #preo atual, n de contratos posicionados,
                                                #ao atual, custo, valor da posio
    valor_cheio = 0.
    ncont_anterior = ncont #salva posio anterior
    ncont += acao #posio atual = pos anterior + ao
    
    if valor != 0:
        valor_cheio = ( valor * ( pmax - pmin ) + pmin )  #valor posicionado atual

    dp = ( preco * ( pmax - pmin ) + pmin ) - valor_cheio #variao do preo atual e do preo de compra/venda
    posicao = ncont_anterior * dp * 10 - custo * abs(acao) #posicao = lucro - custo (INSTANTNEO)

    #calculos sobre o valor    
    if ( ncont_anterior == 0 and acao != 0 ): #primeiro valor
        valor = preco
    elif  ( ncont == 0 ):
        valor = 0
    #caso nao se encaixe nessas condies: valor = valor (nada muda)

    return ncont, valor, posicao, ncont_anterior

def obter_acao(ncont, valores_ant):
    decisao = modelo.toma_acao(valores_ant, teste) #calcula a saida da rede neural

    if decisao == 0: #comprar
        if ncont == 0: #s compra se no tem nada ainda
            return 1
    elif decisao == 1: #vender
        if ncont == 1: #s vende se tiver alguma coisa
            return -1
    return 0 #neutro

def rodar_1dia(precos, custo, dia):
    global melhor_reward
    ncont = 0 #cria variavel de quantidade de contratos
    ncont_anterior = 0 #cria variavel para quantidade de contratos anterior
    valor = 0 #cria variavel para preo medio
    reward = 0 #cria variavel para recompensa
    posicao = 0 #cria variavel de posio 
    modelo.limpa_memoria()

    for step in range( steps[ dia - 1 ], steps[dia] ):  #roda os dados
        
        ultimos_precos = precos[ step : step + 1 ] #pega os valores de agora
        modelo.state = np.append( modelo.state, ultimos_precos ) #adiciona na variavel de estado
        modelo.tira_ultimo_state()
        valores_ant = [ncont, valor, posicao] #grava os valores de antes

        if modelo.state.shape[0] == janela * n_variaveis: #se ja tem memoria suficiente
            acao = obter_acao( ncont, valores_ant ) #obtem ao
            ncont, valor, posicao, ncont_anterior = atuacao(precos['preco'][step], ncont, acao, custo, valor)
            if ( ncont_anterior != ncont ): #reward acumulado recebe reward instantaneo somente se houver lucro/prejuizo real   
                reward += posicao #soma reward
            
        if not teste:
            prox_precos = precos[ step + 1 : step + 2 ] #pega os proximos valores
            modelo.next_state = np.append(modelo.next_state, prox_precos) #adiciona variavel na variavel de proximo estado
            modelo.tira_ultimo_state()
            valores_dps = [ncont, valor, posicao] #grava os valores de depois
            
            if modelo.state.shape[0] == janela * n_variaveis: #se ja tem memoria suficiente
                modelo.treina_modelo(acao, posicao, valores_ant, valores_dps) #roda o modelo
            
    reward += posicao - custo * abs(ncont) #soma reward - DAY-TRADE (obs: custo nao havia sido considerado no reward pq acao era 0)
    if reward > melhor_reward:
        melhor_reward = reward
    return reward #retorna o valor do reward

def rodar_dias(precos, custo):   
    sum_rewards = 0 #cria variavel de somatoria de recompensas
    for dia in range( 1, dias ): #loop de dias
        reward = rodar_1dia(precos, custo, dia)
        sum_rewards += reward #roda 1 dia e adiciona o total na variavel de somatoria
        rewards.append(reward) #guarda o valor do reward
        plotx.append(np.max(plotx) + 1) #guarda o valor do dia
        print("dia: {0}: R$ {1:0.2f}".format(dia, reward)) #mostra o resultado do dia
    return sum_rewards

if __name__ == "__main__":
    sum_rewards_total = 0
    try:
        if teste == True:
            modelo.carrega_pesos('./pesos_treinados_15epocas.h5')
            for t in range(qnt_testes):
                print("teste {0}\n".format(t)) #mostra que epoca vai rodar
                sum_rewards = rodar_dias(inputs, custo) #adiciona o resultado da epoca na somatoria
                sum_rewards_total += sum_rewards
                print("resultado do teste {0} = {1:0.2f}".format(t, sum_rewards))
        else:
            for epoca in range(epocas): #rodar uma quantidade de epocas
                print("epoca {0}\n".format(epoca)) #mostra que epoca vai rodar
                sum_rewards = rodar_dias(inputs, custo) #adiciona o resultado da epoca na somatoria
                sum_rewards_total += sum_rewards
                print("resultado da epoca {0} = {1:0.2f}".format(epoca, sum_rewards))
                modelo.epsilon -= epsilon_decay
    finally:
        if teste != True:
            modelo.salva_pesos('./pesos.h5')
        print("Somatoria dos rewards: {0:0.2f}".format(sum_rewards_total))
        print("Melhor resultado diario: {0:0.2f}".format(melhor_reward))
        plt.plot(plotx, rewards) #plota os valores de reward por dia
