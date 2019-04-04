import pandas as pd #mexer com tabelas
import glob, os #mexer com arquivos
import time #mexer com tempo


HDexterno = "H:/" #coloquei isso porque fica mudando o nome do diretorio
inicioTotal = time.time() #variavel pra contar o tempo


index = ['data_sessao', 'simbolo_instrumento', 'preco_negocio', 'quantidade_negociada', 'horario_negociacao',
                 'indicador_anulacao', 'indicador_direto', 'corretora_compra', 'corretora_venda'] #todas as colunas


arquivoConsolidado = pd.DataFrame(columns=index) #cria variavel consolidada


os.chdir(HDexterno + "TCC/ArquivosFiltrados/v2") #caminho da pasta de importação
for file in glob.glob("*.csv"): #pega arquivos com final CSV
    
    
    inicioParcial = time.time() #variavel para contar o tempo do arquivo
    
    
    arquivoConsolidado = arquivoConsolidado.append(pd.read_csv(file)) #adiciona o novo arquivo no final do arquivo consolidado
    
    
    fim = time.time() - inicioParcial #contabiliza o tempo do arquivo
    print("Arquivo do dia " + file[11:19] + " finalizado em {:.3f}".format(fim) + " segundos") #mostra o tempo que demorou pra filtrar o arquivo
    

print("Tratando da coluna do simbolo")
arquivoConsolidado['simbolo_instrumento'] = arquivoConsolidado['simbolo_instrumento'].str.strip() #trata a coluna do simbolo


arquivoConsolidado.to_csv(HDexterno + "TCC/ArquivoFinal/Consolidado.csv", index=None, header=True) #exporta arquivo do dia WDO


fim = time.time() - inicioTotal #contabiliza o horario
print("Tempo total = {:.3f}".format(fim)) #mostra o tempo que demorou no console