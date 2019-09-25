import numpy as np
import pandas as pd
import DQNModel as dqn
import matplotlib.pyplot as plt
import random

##################### INICIALIZAO DE VARIAVEIS ################################################
steps = [] # 9h04 -> 17h50 a cada 5 segundos 
epocas = 10000 #quantidade de vezes que vai rodar todos os dias
janela = 10 #janela de valores
#n_variaveis = 8 #'preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min', 'IND', 'ISP'
n_variaveis = 1 #'preco'
n_entradas = n_variaveis * janela + 3 #ncont, valor, posicao e inputs
n_neuronios = 216 #numero de neuronios da camada escondida
n_saidas = 3 #nmero de saidas da rede (compra, vende, segura)
custo = 1.06/2 #custo da operao
melhor_reward = 0
posicao_max = 100 #define variavel para normalizar a posicao

versao_arquivo = 2
caminho_arquivo = "C:/Users/Odete/Desktop/consolidado.csv"
#caminho_arquivo = "./consolidado.csv" #caminho para o arquivo de inputs
#index_arquivo = ['preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min', 'IND', 'ISP'] #index do arquivo
index_arquivo = ['preco'] #index do arquivo

carregar_pesos = True
carregar_epoca_epsilon = True
epoca_init = 0
if carregar_epoca_epsilon:
    file = open("./epoca_epsilon.txt", "r")
    valores = file.read().split(',')
    epoca_init = int(valores[0])
    epsilon = float(valores[1])
    file.close()
else:
    epsilon = 1.0 #valor de epsilon
epsilon_min = 0.0001 #valor minimo de epsilon
epsilon_decay = (epsilon - epsilon_min) / (epocas - epoca_init) #o valor que vai retirado do epsilon por epoca

rewards = [0] #variavel para guardar rewards
plotx = [0] #variavel para guardar valores a serem plotados do eixo x

usar_cpu_gpu = True
CPU_cores = 8  # If CPU, how many cores
GPU_mem_use = 1.0  # In both cases the GPU mem is going to be used, choose fraction to use

####################### LEITURA DOS DADOS #######################################################
arquivo = pd.read_csv(caminho_arquivo) #le arquivo
inputs = arquivo[index_arquivo]
if versao_arquivo == 1: #se quiser usar apenas os dias com IND e ISP
    inputs = inputs[inputs['IND'] != 0]
    arquivo = arquivo[arquivo['IND'] != 0]
pmax = np.amax( inputs.loc[:, inputs.columns[0]] ) #define valor minimo do preo
pmin = np.amin( inputs.loc[:, inputs.columns[0]] ) #define valor maximo do preo

for i in range( inputs.shape[1] ): #roda normalizo para todas as colunas
    imax = np.amax( inputs.loc[:, inputs.columns[i]] ) #pega valor maximo
    imin = np.amin( inputs.loc[:, inputs.columns[i]] ) #pega valor minimo
    inputs.loc[:, inputs.columns[i]] = ( inputs.loc[:, inputs.columns[i]] - imin ) / ( imax - imin ) #normaliza prs

dt = arquivo['dt'].values #cria coluna apenas dos dias
steps = []
ultimo_dia = 0
dias_para_rodar = [] #variavel para colocar os dias a serem rodados
j = 0
for i in range( 0, len(dt) ):
    if (dt[i] != ultimo_dia):
        steps.append(i) #numero de linhas entre dias
        ultimo_dia = dt[i]
        dias_para_rodar.append(j) #numero do dia
        j += 1
dias = len(steps)
dias = 100

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
    elif ( ncont == 0 ):
        valor = 0
    #caso nao se encaixe nessas condies: valor = valor (nada muda)

    return ncont, valor, posicao, ncont_anterior

def obter_acao(ncont, valores_ant):
    decisao = modelo.toma_acao(valores_ant, False) #calcula a saida da rede neural

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
    modelo.limpa_memoria() #limpa o vetor de memoria

    dia_para_rodar = dias_para_rodar[dia] #pega um dia random para rodar
    for step in range( steps[dia_para_rodar - 1], steps[dia_para_rodar] ):  #roda os dados
        
        ultimos_precos = precos[ step : step + 1 ] #pega os valores de agora
        modelo.state = np.append( modelo.state, ultimos_precos ) #adiciona na variavel de estado
        modelo.tira_ultimo_state()
        valores_ant = [ncont, valor, posicao / posicao_max] #grava os valores de antes

        if modelo.state.shape[0] == janela * n_variaveis: #se ja tem memoria suficiente
            acao = obter_acao( ncont, valores_ant ) #obtem ao
            ncont, valor, posicao, ncont_anterior = atuacao(precos['preco'][step], ncont, acao, custo, valor)
            if ( ncont_anterior != ncont ): #reward acumulado recebe reward instantaneo somente se houver lucro/prejuizo real   
                reward += posicao #soma reward
            
        prox_precos = precos[ step + 1 : step + 2 ] #pega os proximos valores
        modelo.next_state = np.append(modelo.next_state, prox_precos) #adiciona variavel na variavel de proximo estado
        modelo.tira_ultimo_state()
        valores_dps = [ncont, valor, posicao / posicao_max] #grava os valores de depois
        
        if modelo.state.shape[0] == janela * n_variaveis: #se ja tem memoria suficiente
            modelo.treina_modelo(acao, posicao / posicao_max, valores_ant, valores_dps) #roda o modelo
            
    reward += posicao - custo * abs(ncont) #soma reward - DAY-TRADE (obs: custo nao havia sido considerado no reward pq acao era 0)
    if reward > melhor_reward:
        melhor_reward = reward
    return reward #retorna o valor do reward

def rodar_dias(precos, custo):   
    sum_rewards = 0 #cria variavel de somatoria de recompensas
    random.shuffle(dias_para_rodar) #randomiza vetor de dias
    for dia in range( 1, dias ): #loop de dias
        reward = rodar_1dia(precos, custo, dia)
        sum_rewards += reward #roda 1 dia e adiciona o total na variavel de somatoria
        rewards.append(reward) #guarda o valor do reward
        plotx.append(np.max(plotx) + 1) #guarda o valor do dia
        print("dia {0} de {1}: R$ {2:0.2f}".format(dia, dias, reward)) #mostra o resultado do dia
    return sum_rewards

# =============================================================================
# def config_hard():
#     """
#     Configuration of CPU and GPU
#     """
#     config = tf.ConfigProto()
#     config = tf.ConfigProto(device_count = {'GPU': 0},
#                                 intra_op_parallelism_threads=CPU_cores,
#                                 inter_op_parallelism_threads=CPU_cores)
#     config.gpu_options.per_process_gpu_memory_fraction = GPU_mem_use
#     set_session(tf.Session(config=config))
# =============================================================================

if __name__ == "__main__":
# =============================================================================
#     if usar_cpu_gpu:
#         config_hard()
# =============================================================================
    sum_rewards_total = 0
    try:
        if carregar_pesos:
            modelo.carrega_pesos('./pesos.h5')
        epoca_parou = epoca_init
        for epoca in range(epoca_init, epocas): #rodar uma quantidade de epocas
            print("epoca {0}\n".format(epoca)) #mostra que epoca vai rodar
            sum_rewards = rodar_dias(inputs, custo) #adiciona o resultado da epoca na somatoria
            sum_rewards_total += sum_rewards
            print("resultado da epoca {0} = {1:0.2f}".format(epoca, sum_rewards))
            epoca_parou += 1
            modelo.epsilon -= epsilon_decay
    finally:
        modelo.salva_pesos('./pesos.h5')
        if carregar_epoca_epsilon:
            file = open("./epoca_epsilon.txt", "w")
            file.writelines("{0},{1}".format(epoca_parou, modelo.epsilon))
            file.close()
            print("parou na epoca {0} com epsilon {1}".format(epoca_parou, modelo.epsilon))
        print("Somatoria dos rewards: {0:0.2f}".format(sum_rewards_total))
        print("Melhor resultado diario: {0:0.2f}".format(melhor_reward))
        plt.plot(plotx, rewards) #plota os valores de reward por dia