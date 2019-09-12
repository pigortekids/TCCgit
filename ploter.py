import pandas as pd
import matplotlib.pyplot as plt
import random

def tempoIntToStr(tempo): #função para transformar int para horario
    milesimo = 1000
    horas = int(tempo//(3600 * milesimo))
    tempo -= horas * (3600 * milesimo)
    minutos = int(tempo//(60 * milesimo))
    tempo -= minutos * (60 * milesimo)
    segundos = int(tempo//milesimo)
    tempo -= segundos * milesimo
    return '{:02d}'.format(horas) + ":" + '{:02d}'.format(minutos) + ":" + '{:02d}'.format(segundos) + "." + '{:03d}'.format(tempo)

versao_arquivo = 1
caminho_arquivo = "C:/Users/Odete/Desktop/consolidado.csv"
index_arquivo = ['dt', 'preco', 'hr_int', 'IND'] #index do arquivo
arquivo = pd.read_csv(caminho_arquivo) #le arquivo
arquivo = arquivo[index_arquivo]
if versao_arquivo == 1: #se quiser usar apenas os dias com IND e ISP
    arquivo = arquivo[arquivo['IND'] != 0]
    
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

tempo = [] #variavel para guardar rewards
valor = [] #variavel para guardar valores a serem plotados do eixo x

dia = random.randint(0, len(dias_para_rodar))
plt.title("Dia {0}".format(arquivo.iloc[steps[dia]][0]))
for step in range( steps[ dia - 1 ], steps[dia] ):
    print(arquivo.iloc[step][2])
    tempo.append(arquivo.iloc[step][2])
    valor.append(arquivo.iloc[step][1])
    #plt.scatter(arquivo.iloc[step][2], arquivo.iloc[step][1])
    plt.plot(tempo, valor)
    plt.pause(0.1)
    
plt.show()