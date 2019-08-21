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

def find_nearest_idx(array, value): # função para achar o valor mais perto em um array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
#######################################################################



############################   VARIAVEIS   ############################
HDexterno = "C:/Users/igora/Desktop/TCC/" #coloquei isso porque fica mudando o nome do diretorio
versao = "v7" #versão que esta sendo usada
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

vencimentoWDO = pd.read_csv(pastaCoisas + "vencimentoWDO.csv") #le o arquivo contendo os vencimentos do WDO
feriados = pd.read_csv(pastaCoisas + "feriadosBR.csv") #le o arquivo com os feriados BRs
holidays = feriados['Data'] #pega só as datas
cal = Calendar(holidays=holidays, weekdays=['Sunday', 'Saturday']) #adiciona no calendario para poder controlar

periodicidade = 5 * 1000 #define a periodicidade em que o arquivo vai ser filtrado
horarioInicio = '09:04:00.000' #define horario de inicio para começar a pegar os arquivos
horarioInicioInt = tempoStrToInt(horarioInicio) #traduz o horario inicial
horarioFim = '18:00:00.000' #define horario de fim para terminar de pegar os arquivos
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
    arquivo['simbolo'] = arquivo['simbolo'].str.strip() #tira os espaços em branco
    mascara = arquivo['simbolo'].str.len() == 6 #criando uma mascara para filtrar
    arquivo = arquivo.loc[mascara] #pega so os que tem 6 digitos
    
    
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
    arquivo = arquivo[arquivo['simbolo'].str.contains('WDO' + letraCorreta)] #filtra o arquivo por WDO e a letra do vencimento
    del arquivo['simbolo'] #dleta a coluna do simbolo
    
    
    ################### filtro de horario ###################
    arquivo = arquivo[arquivo['hr'] > horarioInicio] #filtra para começar apenas em um horario certo
    arquivo['hr_int'] = arquivo.apply(lambda row: tempoStrToInt(row['hr']), axis=1) #cria coluna do valor inteiro do horario
    del arquivo['hr'] #deleta a coluna do horario
    arquivo = arquivo.sort_values(by=['hr_int']) #ordena os valores pelo horário
    arquivo.reset_index(drop=True, inplace=True) #reseta o index do arquivo
    arquivoFinal = pd.DataFrame(columns=indexFiltrado2) #cria variavel para consolidar os valores mais perto
    last_idx = 0 #define variavel pro ultimo index que foi pego
    horarioComecar = horarioInicioInt #inicia variavel de hora para começar a rodar o filtro
    while horarioComecar < arquivo.iloc[0, 3]: #roda ate o horario começar seja maior que o primeiro horario do arquivo
        horarioComecar += periodicidade #aumenta o horario começar pela periodicidade
    for horaReferencia in range(horarioComecar + periodicidade, horarioFimInt, periodicidade): #desde a hora de inicio ate a hora final com step de periodicidade
        idx = find_nearest_idx(arquivo['hr_int'], horaReferencia) #index da linha do valor mais perto
        if arquivo.iloc[idx, 3] > horaReferencia: #se o horario for acima do referencia
            idx -= 1 #volta 1
        preco_pon = float(0) #inicia variavel preço ponderado
        qnt_soma = int(0) #inicia variavel somatoria da quantidade
        maximo = arquivo.iloc[idx, 1] #inicia variavel maximo com o valor de agora
        minimo = arquivo.iloc[idx, 1] #inicia variavel minimo com o valor de agora
        for i in range( last_idx , idx + 1 ):
            if maximo < arquivo.iloc[i, 1]: # 1 = preco
                maximo = arquivo.iloc[i, 1]
            if minimo > arquivo.iloc[i, 1]:
                minimo = arquivo.iloc[i, 1]
            preco_pon += arquivo.iloc[i, 1] * arquivo.iloc[i, 2] # 2 = qnt
            qnt_soma += arquivo.iloc[i, 2]
        last_idx = idx + 1 #atualiza a variavel de ultimo index
        if qnt_soma != 0: #caso a somatoria da quantidade seja diferente de 0
            preco_pon = round( preco_pon / qnt_soma , 2) #faz o calculo do preço ponderado
        linhaAdicionar = {'dt':arquivo.iloc[idx, 0], 'preco':arquivo.iloc[idx, 1], 'hr_int':horaReferencia,
                          'preco_pon':preco_pon, 'qnt_soma':qnt_soma, 'max':maximo, 'min':minimo,
                          'IND':0, 'ISP':0} #cria linha a ser adicionada no arquivo final
        arquivoFinal = arquivoFinal.append(linhaAdicionar, ignore_index=True) #adiciona no arquivo final
    
    
    arquivoFinal.to_csv(pastaArquivosFiltrados + "FiltradoDia" + diaDoArquivo + ".csv", index=None, header=True) #exporta arquivo do dia WDO
        
    
    arquivoFinal = None #limpa arquivo final
    arquivo = None #limpa a variavel arquivo
    
    
    #shutil.move(pastaArquivosDescompactados + file, pastaArquivosDescompactadosJaRodados) #move o arquivo para a pasta de ja rodados
    
    
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
    
    
    #shutil.move(pastaArquivosFiltrados + file, pastaArquivosJaConsolidados) #move o arquivo para a pasta de ja consolidados
    
    
    fim = time.time() - inicioParcial #contabiliza o tempo do arquivo
    print("Arquivo do dia " + file[11:19] + " consolidado em {:.3f}".format(fim) + " segundos") #mostra o tempo que demorou pra filtrar o arquivo


arquivoConsolidado.to_csv(pastaArquivoConsolidado + "Consolidado.csv", index=None, header=True) #exporta arquivo do dia WDO
#######################################################################





fim = time.time() - inicioTotal #contabiliza o horario
print("Tempo total = {:.3f}".format(fim)) #mostra o tempo que demorou no console