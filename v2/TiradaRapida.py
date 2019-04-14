import pandas as pd #mexer com tabelas

fileOriginal = "H:/TCC/ArquivoFinal/v3/Consolidado.csv"
fileDestino = "H:/TCC/ArquivoFinal/v4/Consolidado.csv"
arquivo = pd.read_csv(fileOriginal) #le o arquivo
del arquivo['horario_negociacao']
del arquivo['simbolo_instrumento']
arquivo = arquivo.astype({'corretora_compra': int, 'corretora_venda': int, 'indicador_anulacao': bool,
                              'indicador_direto': bool, 'quantidade_negociada': int}) #transforma as colunas
arquivo.to_csv(fileDestino, index=None, header=True) #exporta arquivo do dia WDO