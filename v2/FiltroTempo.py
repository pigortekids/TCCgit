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


inicio = time.time() #variavel para contar o tempo do arquivo


#caminho = "H:/TCC/ArquivoFinal/v2/Consolidado.csv"
caminho = "G:/TCC/ArquivosFiltrados/v2/FiltradoDia20180702.csv"
periodicidade = 2 * 1000 #define a periodicidade em que o arquivo vai ser filtrado
horarioInicio = '09:04:00.000' #define horario de inicio para começar a pegar os arquivos
horarioInicioInt = tempoStrToInt(horarioInicio) #traduz o horario inicial
horarioFim = '17:59:59.000' #define horario de fim para terminar de pegar os arquivos
horarioFimInt = tempoStrToInt(horarioFim) #traduz o horario fim


arquivo =