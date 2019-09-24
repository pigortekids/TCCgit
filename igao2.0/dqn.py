import numpy as np
from env import StockTradingEnv

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, PReLU
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import pandas as pd

arquivo = pd.read_csv("C:/Users/Odete/Desktop/consolidado.csv")
arquivo_comecos = []
ultimo_dia = ''
for i in range( 0, arquivo.shape[0] ):
    if (arquivo.iloc[i]['dt'] != ultimo_dia):
        arquivo_comecos.append(i) #numero de linhas entre dias
        ultimo_dia = arquivo.iloc[i]['dt']

teste = False
passos = 10
memoria = 50000
learning_rate = 1e-4
epocas = 500000
batch_size = 16
NODES = 16  # Neurons

# Get the environment and extract the number of actions.
env = StockTradingEnv(arquivo, arquivo_comecos, passos)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(NODES))
model.add(PReLU())
model.add(Dense(NODES * 2))
model.add(PReLU())
model.add(Dense(NODES * 4))
model.add(PReLU())
model.add(Dense(NODES * 2))
model.add(PReLU())
model.add(Dense(nb_actions))
model.add(Activation('linear'))

memory = SequentialMemory(limit=memoria, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               batch_size=batch_size, target_model_update=1e-2, policy=policy,
               enable_double_dqn=True)
dqn.compile(Adam(lr=learning_rate), metrics=['mae'])

if not teste:
    dqn.fit(env, nb_steps=epocas, visualize=False, verbose=1)
    dqn.save_weights('dqn_weights.h5f', overwrite=True)
else:
    dqn.load_weights('dqn_weights_1.h5f')

dqn.test(env, nb_episodes=50, visualize=False)