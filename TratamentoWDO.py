import pandas as pd #mexer com tabelas
import glob, os, shutil #mexer com arquivos
import time #mexer com tempo
from bizdays import Calendar #usar feriados
import numpy as np #mexer com arrays

###########################   FUNÇÕES   ##############################
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
#######################################################################



############################   VARIAVEIS   ############################
HDexterno = "H:/TCC/" #coloquei isso porque fica mudando o nome do diretorio
versao = "v6" #versão que esta sendo usada
pastaCoisas = HDexterno + "Coisas/" #pasta com arquivos com feriados e vencimentos
pastaArquivosDescompactados = HDexterno + "ArquivosDescompactados/" #pasta dos arquivos originais
pastaArquivosDescompactadosJaRodados = HDexterno + "ArquivosDescompactados/JaRodados/" #pasta dos arquivos originais ja rodados
pastaArquivosFiltrados = HDexterno + "ArquivosFiltrados/" + versao + "/" #pasta dos arquivos já filtrados
pastaArquivoConsolidado = HDexterno + "ArquivoFinal/" + versao + "/" #pasta final do arquivo consolidado
pastaArquivosJaConsolidados = HDexterno + "ArquivosFiltrados/" + versao + "/JaConsolidados/" #pasta dos arquivos ja consolidados
inicioTotal = time.time() #variavel pra contar o tempo

indexTotal = ['data_sessao', 'simbolo_instrumento', 'numero_negocio', 'preco_negocio',
        'quantidade_negociada', 'horario_negociacao', 'indicador_anulacao',
        'data_oferta_compra', 'numero_seq_compra', 'numero_geracao_compra', 'codigo_id_compra',
        'data_oferta_venda', 'numero_seq_venda', 'numero_geracao_venda', 'codigo_id_venda',
        'indicador_direto', 'corretora_compra', 'corretora_venda'] #todos as colunas do arquivo
indexFiltrado = ['data_sessao', 'simbolo_instrumento', 'preco_negocio', 'horario_negociacao'] #primeiro filtro de colunas
indexFiltrado2 = ['data_sessao', 'preco_negocio', 'horario_negociacao'] #segundo filtro de colunas

vencimentoWDO = pd.read_csv(pastaCoisas + "vencimentoWDO.csv") #le o arquivo contendo os vencimentos do WDO
feriados = pd.read_csv(pastaCoisas + "feriadosBR.csv") #le o arquivo com os feriados BRs
holidays = feriados['Data'] #pega só as datas
cal = Calendar(holidays=holidays, weekdays=['Sunday', 'Saturday']) #adiciona no calendario para poder controlar

periodicidade = 2 * 1000 #define a periodicidade em que o arquivo vai ser filtrado
horarioInicio = '09:04:00.000' #define horario de inicio para começar a pegar os arquivos
horarioInicioInt = tempoStrToInt(horarioInicio) #traduz o horario inicial
horarioFim = '18:00:01.000' #define horario de fim para terminar de pegar os arquivos
horarioFimInt = tempoStrToInt(horarioFim) #traduz o horario fim
#######################################################################





##############################   FILTRO   #############################
os.chdir(pastaArquivosDescompactados) #caminho da pasta de importação
for file in glob.glob("*.txt"): #pega arquivos com final TXT
    
    
    inicioParcial = time.time() #variavel para contar o tempo do arquivo
    
    
    #################### filtros basicos ####################
    arquivo = pd.read_csv(file, sep=';', skiprows=1, header=None, names=indexTotal) #le o arquivo
    arquivo.drop(arquivo.tail(1).index, inplace=True) #tira ultima linha do arquivo
    arquivo = arquivo.filter(items=indexFiltrado) #pega apenas as colunas que precisamos
    arquivo['simbolo_instrumento'] = arquivo['simbolo_instrumento'].str.strip() #tira os espaços em branco
    
    
    #################### filtro do WDO e vencimento ####################
    diaDoArquivo = file[8:16] #pega a data do arquivo
    mesDoDia = int(diaDoArquivo[4:6]) #pega o mes da data do movimento
    anoDoDia = int(diaDoArquivo[:4]) #pega o ano da data do movimento
    ultimoDiaUtil = cal.getdate('last bizday', anoDoDia, mesDoDia) #pega o ultimo dia util do mes da data do movimento
    mesCorreto = mesDoDia + 1 #define uma variavel de mes correto adicionando um no mes do dia do movimento
    anoCorreto = anoDoDia #define uma variavel de ano correto
    if str(ultimoDiaUtil).replace('-', '') == str(diaDoArquivo): #confere se é o ultimo dia util do mes
        mesCorreto += 1 #adiciona um na variavel do mes correto
    if mesCorreto > 12: #se o mes passar de 12
        mesCorreto -= 12 #tira 12 do valor do mes
        anoCorreto += 1 #e aumenta em um o ano
    letraCorreta = vencimentoWDO[vencimentoWDO['mes'] == mesCorreto].iloc[0, 0] + str(anoCorreto)[2:] #define a letra que deveria estar no codigo
    arquivo = arquivo[arquivo['simbolo_instrumento'].str.contains('WDO' + letraCorreta)] #filtra o arquivo por WDO e a letra do vencimento
    del arquivo['simbolo_instrumento'] #dleta a coluna do simbolo
    
    
    ################### filtro de horario ###################
    arquivo = arquivo[arquivo['horario_negociacao'] > horarioInicio] #filtra para começar apenas em um horario certo
    arquivo['horario_int'] = arquivo.apply(lambda row: tempoStrToInt(row['horario_negociacao']), axis=1) #cria coluna do valor inteiro do horario
    del arquivo['horario_negociacao'] #deleta a coluna do horario
    colunaHorarios = np.array(arquivo['horario_int'].values.tolist()) #cria coluna de referencia dos horarios do arquivo
    arquivoFinal = pd.DataFrame(columns=arquivo.columns) #cria variavel para consolidar os valores mais perto
    for horaReferencia in range(horarioInicioInt, horarioFimInt, periodicidade): #desde a hora de inicio ate a hora final com step de periodicidade
        maisPerto = find_nearest(colunaHorarios, horaReferencia) #acha o horario mais perto
        idx = find_nearest_idx(colunaHorarios, horaReferencia) #index da linha do valor mais perto
        linhaAdicionar = arquivo.iloc[idx] #cria variavel da linha a ser adicionada
        arquivoFinal = arquivoFinal.append(linhaAdicionar) #adiciona no arquivo final
    
    
    arquivoFinal.to_csv(pastaArquivosFiltrados + "FiltradoDia" + diaDoArquivo + ".csv", index=None, header=True) #exporta arquivo do dia WDO
        
    
    colunaHorarios = None #limpa coluna de referencia
    arquivoFinal = None #limpa arquivo final
    arquivo = None #limpa a variavel arquivo
    
    
    shutil.move(pastaArquivosDescompactados + file, pastaArquivosDescompactadosJaRodados) #move o arquivo para a pasta de ja rodados
    
    
    fim = time.time() - inicioParcial #contabiliza o tempo do arquivo
    print("Arquivo do dia " + diaDoArquivo + " filtrado em {:.3f}".format(fim) + " segundos") #mostra o tempo que demorou pra filtrar o arquivo
#######################################################################





#############################   CONSOLIDADO   #########################
arquivoConsolidado = pd.DataFrame(columns=indexFiltrado2) #cria variavel consolidada
if os.path.exists(pastaArquivoConsolidado + "Consolidado.csv"): #confere se ja existe um arquivo de consolidado
    arquivoConsolidado = arquivoConsolidado.append(pd.read_csv(pastaArquivoConsolidado + "Consolidado.csv")) #adiciona os valor do arquivo consolidado na variável


os.chdir(pastaArquivosFiltrados) #caminho da pasta de importação
for file in glob.glob("*.csv"): #pega arquivos com final CSV
    
    
    inicioParcial = time.time() #variavel para contar o tempo do arquivo
    
    
    arquivoConsolidado = arquivoConsolidado.append(pd.read_csv(file)) #adiciona o novo arquivo no final do arquivo consolidado
    
    
    shutil.move(pastaArquivosFiltrados + file, pastaArquivosJaConsolidados) #move o arquivo para a pasta de ja consolidados
    
    
    fim = time.time() - inicioParcial #contabiliza o tempo do arquivo
    print("Arquivo do dia " + file[11:19] + " consolidado em {:.3f}".format(fim) + " segundos") #mostra o tempo que demorou pra filtrar o arquivo


arquivoConsolidado.to_csv(pastaArquivoConsolidado + "Consolidado.csv", index=None, header=True) #exporta arquivo do dia WDO
#######################################################################





fim = time.time() - inicioTotal #contabiliza o horario
print("Tempo total = {:.3f}".format(fim)) #mostra o tempo que demorou no console