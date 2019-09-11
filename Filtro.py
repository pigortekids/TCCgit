import pandas as pd #mexer com tabelas
import glob, os, shutil #mexer com arquivos
import time #mexer com tempo
from bizdays import Calendar #usar feriados
import numpy as np #mexer com arrays
import datetime

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

def find_nearest_idx(array, value): # função para achar o valor mais perto em um array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
#######################################################################



############################   VARIAVEIS   ############################
HDexterno = "H:/TCC/" #coloquei isso porque fica mudando o nome do diretorio
versao = "v9" #versão que esta sendo usada
pastaCoisas = HDexterno + "Coisas/" #pasta com arquivos com feriados e vencimentos
pastaArquivosDescompactados = HDexterno + "ArquivosDescompactados/" #pasta dos arquivos originais
pastaArquivosDescompactadosJaRodados = HDexterno + "ArquivosDescompactados/JaRodados/" #pasta dos arquivos originais ja rodados
pastaArquivosFiltrados = HDexterno + "ArquivosFiltrados/" + versao + "/" #pasta dos arquivos já filtrados
pastaArquivoConsolidado = HDexterno + "ArquivoFinal/" + versao + "/" #pasta final do arquivo consolidado
pastaArquivosJaConsolidados = HDexterno + "ArquivosFiltrados/" + versao + "/JaConsolidados/" #pasta dos arquivos ja consolidados
inicioTotal = time.time() #variavel pra contar o tempo

indexTotal = ['dt', 'simbolo', 'numero_negocio', 'preco', 'qnt', 'hr', 'indicador_anulacao',
        'data_oferta_compra', 'numero_seq_compra', 'numero_geracao_compra', 'codigo_id_compra',
        'data_oferta_venda', 'numero_seq_venda', 'numero_geracao_venda', 'codigo_id_venda',
        'indicador_direto', 'corretora_compra', 'corretora_venda'] #todos as colunas do arquivo
indexFiltrado = ['dt', 'simbolo', 'preco', 'qnt', 'hr'] #primeiro filtro de colunas
indexFiltrado2 = ['dt', 'preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min', 'IND', 'ISP'] #segundo filtro de colunas

vencimento = pd.read_csv(pastaCoisas + "vencimento.csv") #le o arquivo contendo os vencimentos do WDO
feriados = pd.read_csv(pastaCoisas + "feriadosBR.csv") #le o arquivo com os feriados BRs
holidays = feriados['Data'] #pega só as datas
cal = Calendar(holidays=holidays, weekdays=['Sunday', 'Saturday']) #adiciona no calendario para poder controlar

periodicidade = 5 * 60 * 1000 #define a periodicidade em que o arquivo vai ser filtrado
horarioInicio = '09:05:00.000' #define horario de inicio para começar a pegar os arquivos
horarioInicioInt = tempoStrToInt(horarioInicio) #traduz o horario inicial
horarioFim = '17:55:00.000' #define horario de fim para terminar de pegar os arquivos
horarioFimInt = tempoStrToInt(horarioFim) #traduz o horario fim
#######################################################################





##############################   FILTRO   #############################
os.chdir(pastaArquivosDescompactados) #caminho da pasta de importação
qnt_arquivos = len(glob.glob("*.txt"))
arquivo_n = 0
for file in glob.glob("*.txt"): #pega arquivos com final TXT
    
    
    inicioParcial = time.time() #variavel para contar o tempo do arquivo
    
    
    #################### filtros basicos ####################
    arquivo = pd.read_csv(file, sep=';', skiprows=1, header=None, names=indexTotal) #le o arquivo
    arquivo.drop(arquivo.tail(1).index, inplace=True) #tira ultima linha do arquivo
    arquivo = arquivo.filter(items=indexFiltrado) #pega apenas as colunas que precisamos
    arquivo['simbolo'] = arquivo['simbolo'].str.strip() #tira os espaços em branco
    mascara = arquivo['simbolo'].str.len() == 6 #criando uma mascara para filtrar
    arquivo = arquivo.loc[mascara] #pega so os que tem 6 digitos
    
    
    #################### filtro do WDO ####################
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
    letraCorreta = vencimento[vencimento['mes'] == mesCorreto].iloc[0, 0] + str(anoCorreto)[2:] #define a letra que deveria estar no codigo
    arquivoWDO = arquivo[arquivo['simbolo'].str.contains('WDO' + letraCorreta)] #filtra o arquivo por WDO e a letra do vencimento
    del arquivoWDO['simbolo'] #dleta a coluna do simbolo
    
    #################### filtro ISP ####################
    diaCorreto = datetime.date(int(diaDoArquivo[:4]), int(diaDoArquivo[4:6]), int(diaDoArquivo[-2:]))
    mesCorreto = mesDoDia
    anoCorreto = anoDoDia
    if mesDoDia % 3 == 0:
        diaDoMes = datetime.date(int(diaDoArquivo[:4]), int(diaDoArquivo[4:6]), 1)
        qnt_sexta = 0
        while qnt_sexta < 3:
            if diaDoMes.weekday() == 4: #4 é sexta-feira
                qnt_sexta += 1
            diaDoMes += datetime.timedelta(days=1)
        diaDoMes -= datetime.timedelta(days=2)
        if diaCorreto > diaDoMes:
            mesCorreto += 1
    while mesCorreto % 3 != 0:
        mesCorreto += 1
    if mesCorreto > 12: #se o mes passar de 12
        mesCorreto -= 12 #tira 12 do valor do mes
        anoCorreto += 1 #e aumenta em um o ano
    letraCorreta = vencimento[vencimento['mes'] == mesCorreto].iloc[0, 0] + str(anoCorreto)[2:]
    arquivoISP = arquivo[arquivo['simbolo'].str.contains('ISP' + letraCorreta)] #filtra o arquivo por WDO e a letra do vencimento
    del arquivoISP['simbolo'] #dleta a coluna do simbolo
    
    
    #################### filtro do IND ####################
    diaCorreto = datetime.date(int(diaDoArquivo[:4]), int(diaDoArquivo[4:6]), int(diaDoArquivo[-2:]))
    mesCorreto = mesDoDia
    anoCorreto = anoDoDia
    if mesDoDia % 2 == 0:
        diaDoMes = datetime.date(int(diaDoArquivo[:4]), int(diaDoArquivo[4:6]), 15)
        diaPraCima = diaDoMes
        diferencaCima = 0
        while diaPraCima.weekday() != 2:
            diaPraCima += datetime.timedelta(days=1)
            diferencaCima += 1
        diaPraBaixo = diaDoMes
        diferencaBaixo = 0
        while diaPraBaixo.weekday() != 2:
            diaPraBaixo -= datetime.timedelta(days=1)
            diferencaBaixo += 1
        if diferencaCima < diferencaBaixo:
            diaDoMes = diaPraCima
        else:
            diaDoMes = diaPraBaixo
        diaDoMes -= datetime.timedelta(days=1)
        if diaCorreto > diaDoMes:
            mesCorreto += 1
    while mesCorreto % 2 != 0:
        mesCorreto += 1
    if mesCorreto > 12: #se o mes passar de 12
        mesCorreto -= 12 #tira 12 do valor do mes
        anoCorreto += 1 #e aumenta em um o ano
    letraCorreta = vencimento[vencimento['mes'] == mesCorreto].iloc[0, 0] + str(anoCorreto)[2:]
    arquivoIND = arquivo[arquivo['simbolo'].str.contains('IND' + letraCorreta)] #filtra o arquivo por WDO e a letra do vencimento
    del arquivoIND['simbolo'] #dleta a coluna do simbolo
    
    
    ################### filtro de horario ###################
    arquivoWDO = arquivoWDO[arquivoWDO['hr'] > horarioInicio] #filtra para começar apenas em um horario certo
    arquivoWDO['hr_int'] = arquivoWDO.apply(lambda row: tempoStrToInt(row['hr']), axis=1) #cria coluna do valor inteiro do horario
    del arquivoWDO['hr'] #deleta a coluna do horario
    arquivoWDO = arquivoWDO.sort_values(by=['hr_int']) #ordena os valores pelo horário
    arquivoWDO.reset_index(drop=True, inplace=True) #reseta o index do arquivo
    last_idxWDO = 0 #define variavel pro ultimo index que foi pego
    
    
    arquivoISP = arquivoISP[arquivoISP['hr'] > horarioInicio] #filtra para começar apenas em um horario certo
    arquivoISP['hr_int'] = arquivoISP.apply(lambda row: tempoStrToInt(row['hr']), axis=1) #cria coluna do valor inteiro do horario
    del arquivoISP['hr'] #deleta a coluna do horario
    arquivoISP = arquivoISP.sort_values(by=['hr_int']) #ordena os valores pelo horário
    arquivoISP.reset_index(drop=True, inplace=True) #reseta o index do arquivo
    
    
    arquivoIND = arquivoIND[arquivoIND['hr'] > horarioInicio] #filtra para começar apenas em um horario certo
    arquivoIND['hr_int'] = arquivoIND.apply(lambda row: tempoStrToInt(row['hr']), axis=1) #cria coluna do valor inteiro do horario
    del arquivoIND['hr'] #deleta a coluna do horario
    arquivoIND = arquivoIND.sort_values(by=['hr_int']) #ordena os valores pelo horário
    arquivoIND.reset_index(drop=True, inplace=True) #reseta o index do arquivo
    
    
    arquivoFinal = pd.DataFrame(columns=indexFiltrado2) #cria variavel para consolidar os valores mais perto
    horarioComecar = horarioInicioInt #inicia variavel de hora para começar a rodar o filtro
    horarioMinimo = min([arquivoWDO.iloc[0, 3], arquivoISP.iloc[0, 3], arquivoIND.iloc[0, 3]])
    
    
    for horaReferencia in range(horarioComecar, horarioFimInt, periodicidade):
        idxWDO = find_nearest_idx(arquivoWDO['hr_int'], horaReferencia) #index da linha do valor mais perto
        if arquivoWDO.iloc[idxWDO, 3] > horaReferencia: #se o horario for acima do referencia
            if idxWDO != 0:
                idxWDO -= 1 #volta 1
        preco_pon = float(0) #inicia variavel preço ponderado
        qnt_soma = int(0) #inicia variavel somatoria da quantidade
        maximo = arquivoWDO.iloc[idxWDO, 1] #inicia variavel maximo com o valor de agora
        minimo = arquivoWDO.iloc[idxWDO, 1] #inicia variavel minimo com o valor de agora
        for i in range( last_idxWDO , idxWDO + 1 ):
            if maximo < arquivoWDO.iloc[i, 1]: # 1 = preco
                maximo = arquivoWDO.iloc[i, 1]
            if minimo > arquivoWDO.iloc[i, 1]:
                minimo = arquivoWDO.iloc[i, 1]
            preco_pon += arquivoWDO.iloc[i, 1] * arquivoWDO.iloc[i, 2] # 2 = qnt
            qnt_soma += arquivoWDO.iloc[i, 2]
        last_idxWDO = idxWDO + 1 #atualiza a variavel de ultimo index
        if qnt_soma != 0: #caso a somatoria da quantidade seja diferente de 0
            preco_pon = round( preco_pon / qnt_soma , 2) #faz o calculo do preço ponderado
            
        
        idxISP = find_nearest_idx(arquivoISP['hr_int'], horaReferencia) #index da linha do valor mais perto
        if arquivoISP.iloc[idxISP, 3] > horaReferencia: #se o horario for acima do referencia
            if idxISP != 0:
                idxISP -= 1 #volta 1
                
                
        idxIND = find_nearest_idx(arquivoIND['hr_int'], horaReferencia) #index da linha do valor mais perto
        if arquivoIND.iloc[idxIND, 3] > horaReferencia: #se o horario for acima do referencia
            if idxIND != 0:
                idxIND -= 1 #volta 1
                
        
        linhaAdicionar = {'dt':arquivoWDO.iloc[idxWDO, 0], 'preco':arquivoWDO.iloc[idxWDO, 1], 'hr_int':horaReferencia,
                          'preco_pon':preco_pon, 'qnt_soma':qnt_soma, 'max':maximo, 'min':minimo,
                          'IND':arquivoIND.iloc[idxIND, 1], 'ISP':arquivoISP.iloc[idxISP, 1]} #cria linha a ser adicionada no arquivo final
        arquivoFinal = arquivoFinal.append(linhaAdicionar, ignore_index=True) #adiciona no arquivo final
    
    
    arquivoFinal.to_csv(pastaArquivosFiltrados + "filtro_dia_" + diaDoArquivo + ".csv", index=None, header=True) #exporta arquivo do dia WDO
        
    
    arquivoFinal = None #limpa arquivo final
    arquivo = None #limpa a variavel arquivo
    arquivoWDO = None #limpa a variavel arquivoWDO
    arquivoISP = None #limpa a variavel arquivoISP
    arquivoIND = None #limpa a variavel arquivoIND
    
    
    shutil.move(pastaArquivosDescompactados + file, pastaArquivosDescompactadosJaRodados) #move o arquivo para a pasta de ja rodados
    
    
    arquivo_n += 1
    fim = time.time() - inicioParcial #contabiliza o tempo do arquivo
    print("({0}/{1}) Arquivo do dia {2} filtrado em {3:.3f} segundos".format(arquivo_n, qnt_arquivos, diaDoArquivo, fim)) #mostra o tempo que demorou pra filtrar o arquivo
#######################################################################





#############################   CONSOLIDADO   #########################
arquivoConsolidado = pd.DataFrame(columns=indexFiltrado2) #cria variavel consolidada
if os.path.exists(pastaArquivoConsolidado + "Consolidado.csv"): #confere se ja existe um arquivo de consolidado
    arquivoConsolidado = arquivoConsolidado.append(pd.read_csv(pastaArquivoConsolidado + "Consolidado.csv")) #adiciona os valor do arquivo consolidado na variável


os.chdir(pastaArquivosFiltrados) #caminho da pasta de importação
qnt_arquivos = len(glob.glob("*.csv"))
arquivo_n = 0
for file in glob.glob("*.csv"): #pega arquivos com final CSV
    
    
    inicioParcial = time.time() #variavel para contar o tempo do arquivo
    
    
    arquivoConsolidado = arquivoConsolidado.append(pd.read_csv(file)) #adiciona o novo arquivo no final do arquivo consolidado
    
    
    shutil.move(pastaArquivosFiltrados + file, pastaArquivosJaConsolidados) #move o arquivo para a pasta de ja consolidados
    
    
    arquivo_n += 1
    fim = time.time() - inicioParcial #contabiliza o tempo do arquivo
    print("({0}/{1}) Arquivo do dia {2} consolidado em {3:.3f} segundos".format(arquivo_n, qnt_arquivos, file[11:19], fim)) #mostra o tempo que demorou pra filtrar o arquivo


arquivoConsolidado.to_csv(pastaArquivoConsolidado + "consolidado.csv", index=None, header=True) #exporta arquivo do dia WDO
#######################################################################





fim = time.time() - inicioTotal #contabiliza o horario
print("Tempo total = {:.3f}".format(fim)) #mostra o tempo que demorou no console