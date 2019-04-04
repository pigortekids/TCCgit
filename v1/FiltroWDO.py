import pandas as pd #mexer com tabelas
import glob, os #mexer com arquivos
import time #mexer com tempo
from bizdays import Calendar #usar feriados


HDexterno = "H:/" #coloquei isso porque fica mudando o nome do diretorio
inicioTotal = time.time() #variavel pra contar o tempo


indexTotal = ['data_sessao', 'simbolo_instrumento', 'numero_negocio', 'preco_negocio',
        'quantidade_negociada', 'horario_negociacao', 'indicador_anulacao',
        'data_oferta_compra', 'numero_seq_compra', 'numero_geracao_compra', 'codigo_id_compra',
        'data_oferta_venda', 'numero_seq_venda', 'numero_geracao_venda', 'codigo_id_venda',
        'indicador_direto', 'corretora_compra', 'corretora_venda'] #todos as colunas do arquivo
indexFiltrado = ['data_sessao', 'simbolo_instrumento', 'preco_negocio', 'quantidade_negociada', 'horario_negociacao',
                 'indicador_anulacao', 'indicador_direto', 'corretora_compra', 'corretora_venda'] #todas as colunas que precisamos


vencimentoWDO = pd.read_csv(HDexterno + "TCC/vencimentoWDO.csv") #le o arquivo contendo os vencimentos do WDO
feriados = pd.read_csv(HDexterno + "TCC/feriadosBR.csv") #le o arquivo com os feriados BRs
holidays = feriados['Data'] #pega só as datas
cal = Calendar(holidays=holidays, weekdays=['Sunday', 'Saturday']) #adiciona no calendario para poder controlar


os.chdir(HDexterno + "TCC/ArquivosDescompactados") #caminho da pasta de importação
for file in glob.glob("*.txt"): #pega arquivos com final TXT
    
    
    inicioParcial = time.time() #variavel para contar o tempo do arquivo
    
    
    arquivo = pd.read_csv(file, sep=';', skiprows=1, header=None, names=indexTotal) #le o arquivo
    arquivo.drop(arquivo.tail(1).index, inplace=True) #tira ultima linha do arquivo
    arquivo = arquivo.filter(items=indexFiltrado) #pega apenas as colunas que precisamos
    
    
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
    
    
    arquivoFiltradoWDO = arquivo[arquivo['simbolo_instrumento'].str.contains('WDO' + letraCorreta)] #filtra o arquivo por WDO e a letra do vencimento
    
    
    arquivoFiltradoWDO.to_csv(HDexterno + "TCC/ArquivosFiltrados/v2/FiltradoDia" + diaDoArquivo + ".csv", index=None, header=True) #exporta arquivo do dia WDO
    
    
    arquivo = None #limpa a variavel arquivo
    arquivoFiltradoWDO = None #limpa a variavel filtrada
    
    
    fim = time.time() - inicioParcial #contabiliza o tempo do arquivo
    print("Arquivo do dia " + diaDoArquivo + " finalizado em {:.3f}".format(fim) + " segundos") #mostra o tempo que demorou pra filtrar o arquivo


fim = time.time() - inicioTotal #contabiliza o horario
print("Tempo total = {:.3f}".format(fim)) #mostra o tempo que demorou no console