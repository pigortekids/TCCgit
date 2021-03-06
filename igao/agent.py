from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from IgaoEnv import IgaoEnv
import pandas as pd

WINDOW_LENGTH = 50 #memoria que ele vai olhar
num_actions = 3 #quantidade de ações

df = pd.read_csv('H:/TCC/ArquivoFinal/v6/Consolidado.csv') #le o arquivo de dados
env = IgaoEnv(df) #cria o enviroment pro keras
# =============================================================================
# df_train = 'H:/TCC/Codigos/v4/trading-rl/trading_agent/data/train_data.npy'
# env_train = IgaoEnv(df_train) #cria o enviroment pro keras
# df_test = 'H:/TCC/Codigos/v4/trading-rl/trading_agent/data/test_data.npy'
# env_test = IgaoEnv(df_test) #cria o enviroment pro keras
# =============================================================================

model = Sequential() #modelo do keras
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape)) #camada de entrada
model.add(Dense(16, activation='relu')) #camada escondida
model.add(Dense(num_actions, activation='linear')) #camada de saida
#print(model.summary()) #printa um resumo do modelo

memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH) #cria uma memoria

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.,
                              value_min=.1,
                              value_test=.05,
                              nb_steps=1000000) #cria uma função de seleção das ações

dqn = DQNAgent(model=model,
               nb_actions=num_actions,
               policy=policy,
               memory=memory,
               nb_steps_warmup=50000,
               gamma=.99,
               target_model_update=10000,
               train_interval=4,
               delta_clip=1.) #cria o agente que vai tomar as ações

dqn.compile(Adam(lr=.00025), metrics=['mae']) #compila

caminho = "H:/TCC/Codigos/v3/pesos.h5f" #caminho para salvar os pesos

######################TREINO######################
dqn.fit(env, nb_steps=3500000, visualize=False, verbose=2) #treinamento
dqn.save_weights(caminho, overwrite=True) #salvando os pesos da rede neural
##################################################

#######################TESTE######################
# =============================================================================
# dqn.load_weights(caminho) #pegando pesos ja treinados
# =============================================================================
##################################################

dqn.test(env, nb_episodes=15, visualize=False) #testando o modelo