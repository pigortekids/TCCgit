########################################   BIBLIOTECAS ####################################
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

##################### MODELO DQN ####################################################
class DQNAgent:
    ########################### INICIALIZA ###########################################
    def __init__(self, state_size, action_size, epsilon, janela):
        self.state_size = state_size
        self.action_size = action_size
        self.memoria = deque(maxlen=janela)
        print(self.memoria)
        self.gamma = 0.95    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.learning_rate = 0.001
        self.model = self.cria_modelo()

################################# REDE NEURAL ###########################################
    def cria_modelo(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu')) #camada de entrada (escondida)
        model.add(Dense(64, activation='relu')) #camada escondida
        model.add(Dense(self.action_size, activation='softmax')) #camada de saida
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate)) #compilador
        return model

    def adiciona_memoria(self, state, action, reward, next_state):
        self.memoria.append((state, action, reward, next_state)) #adiciona um valor na memoria

    def limpa_memoria(self, janela):
        self.memoria = deque(maxlen=janela)
        
    def toma_acao(self, state):
        if np.random.rand() <= self.epsilon: #se o numero aleatorio for menor que o epsilon
            return random.randrange(self.action_size) #retorna ação aleatoria
        act_values = self.model.predict(state) #calcula qual a melhor ação
        return np.argmax(act_values[0])  # returns action

    def treina_modelo(self, batch_size):
        minibatch = random.sample(self.memoria, batch_size)
        for state, action, reward, next_state in minibatch:
            target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
    def carrega_pesos(self, name):
        self.model.load_weights(name) #carrega pesos

    def salva_pesos(self, name):
        self.model.save_weights(name) #salva pesos