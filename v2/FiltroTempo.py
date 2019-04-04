import numpy as np #mexer com arrays
import pandas as pd #mexer com tabelas
import time #mexer com tempo

def tempoStrToInt(tempo): #função para transformar horario em int
    milesimo = 1000
    horas = int(tempo[:2]) * 3600 * milesimo
    minutos = int(tempo[3:5]) * 60 * milesimo
    segundos = int(tempo[6:8]) * milesimo
    milesimos = int(tempo[-3:])
    return horas + minutos + segundos + milesimos
    
def tempoIntToStr(tempo): #função para transformar int para horario
    milesimo = 1000
    horas = int(tempo//(3600 * milesimo))
    tempo -= horas * (3600 * milesimo)
    minutos = int(tempo//(60 * milesimo))
    tempo -= minutos * (60 * milesimo)
    segundos = int(tempo//milesimo)
    tempo -= segundos * milesimo
    return '{:02d}'.format(horas) + ":" + '{:02d}'.format(minutos) + ":" + '{:02d}'.format(segundos) + "." + '{:03d}'.format(tempo)
    
def find_nearest(array, value): # função para achar o valor mais perto em um array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value): # função para achar o valor mais perto em um array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


inicio = time.time() #variavel para contar o tempo do arquivo


#caminho = "H:/TCC/ArquivoFinal/v2/Consolidado.csv"
caminho = "H:/TCC/ArquivosFiltrados/v2/FiltradoDia20180702.csv"
periodicidade = 2 * 1000 #define a periodicidade em que o arquivo vai ser filtrado
horarioInicio = '09:04:00.000' #define horario de inicio para começar a pegar os arquivos
horarioInicioInt = tempoStrToInt(horarioInicio) #traduz o horario inicial
horarioFim = '18:00:00.000' #define horario de fim para terminar de pegar os arquivos
horarioFimInt = tempoStrToInt(horarioFim) #traduz o horario fim


arquivo = pd.read_csv(caminho) #le o arquivo
arquivo = arquivo[arquivo['horario_negociacao'] > horarioInicio] #filtra para começar apenas em um horario certo
arquivo['horario_int'] = arquivo.apply(lambda row: tempoStrToInt(row['horario_negociacao']), axis=1) #cria coluna do valor inteiro do horario


colunaHorarios = np.array(arquivo['horario_int'].values.tolist()) #cria coluna de referencia dos horarios do arquivo


arquivoFinal = pd.DataFrame(columns=arquivo.columns) #cria variavel para consolidar os valores mais perto


for horaReferencia in range(horarioInicioInt, horarioFimInt, periodicidade): #desde a hora de inicio ate a hora final com step de periodicidade
    maisPerto = find_nearest(colunaHorarios, horaReferencia) #acha o horario mais perto
    colunaQuantidade = arquivo[arquivo['horario_int'] == maisPerto] #filtra o arquivo só para os casos que tem o mesmo horario
    qntTotal = int(colunaQuantidade['quantidade_negociada'].sum()) #soma todas as quantidades de mesmo horario
    idx = find_nearest_idx(colunaHorarios, horaReferencia) #index da linha do valor mais perto
    linhaAdicionar = arquivo.iloc[idx] #cria variavel da linha a ser adicionada
    linhaAdicionar['quantidade_negociada'] = qntTotal #muda o valor da quantidade para o total
    arquivoFinal = arquivoFinal.append(linhaAdicionar) #adiciona no arquivo final
# =============================================================================
#     if horaReferencia == tempoStrToInt('10:00:00.000'):
#         break
# =============================================================================
# =============================================================================
#     if colunaQuantidade.shape[0] > 1:
#         break
# =============================================================================


fim = time.time() - inicio #contabiliza o horario
print("Tempo total = {:.3f}".format(fim)) #mostra o tempo que demorou no console