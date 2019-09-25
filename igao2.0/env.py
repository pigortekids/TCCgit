import random
import gym
from gym import spaces
import numpy as np

def tempoIntToStr(tempo): #função para transformar int para horario
    milesimo = 1000
    horas = int(tempo//(3600 * milesimo))
    tempo -= horas * (3600 * milesimo)
    minutos = int(tempo//(60 * milesimo))
    tempo -= minutos * (60 * milesimo)
    segundos = int(tempo//milesimo)
    tempo -= segundos * milesimo
    return '{:02d}'.format(horas) + ":" + '{:02d}'.format(minutos) + ":" + '{:02d}'.format(segundos) + "." + '{:03d}'.format(tempo)

colunas = 8

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, comeco_dias, passos):
        super(StockTradingEnv, self).__init__()

        self.passos = passos
        self.df = df
        self.comeco_dias = comeco_dias
        self.max_preco = df['preco'].max()
        self.min_preco = df['preco'].min()
        self.max_hr_int = df['hr_int'].max()
        self.max_preco_pon = df['preco_pon'].max()
        self.max_qnt_soma = df['qnt_soma'].max()
        self.max_max = df['max'].max()
        self.max_min = df['min'].max()
        self.max_IND = df['IND'].max()
        self.max_ISP = df['ISP'].max()
        self.custo = 1.06/2

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Discrete(3)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(colunas, passos), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        steps_atras = self.current_step - self.passos + 1
        obs = np.array([
            self.df.loc[steps_atras: self.current_step, 'preco'].values / self.max_preco,
            self.df.loc[steps_atras: self.current_step, 'hr_int'].values / self.max_hr_int,
            self.df.loc[steps_atras: self.current_step, 'preco_pon'].values / self.max_preco_pon,
            self.df.loc[steps_atras: self.current_step, 'qnt_soma'].values / self.max_qnt_soma,
            self.df.loc[steps_atras: self.current_step, 'max'].values / self.max_max,
            self.df.loc[steps_atras: self.current_step, 'min'].values / self.max_min,
            self.df.loc[steps_atras: self.current_step, 'IND'].values / self.max_IND,
            self.df.loc[steps_atras: self.current_step, 'ISP'].values / self.max_ISP,
        ])

        return obs

    def _take_action(self, action):
        
        valor_cheio = 0
        self.ncont_anterior = self.ncont #salva posio anterior
        acao = action - 1
        if (acao == 1 and self.ncont == 0) or (acao == -1 and self.ncont == 1):
            self.ncont += acao
            
        if self.valor != 0:
            valor_cheio = ( self.valor * ( self.max_preco - self.min_preco ) + self.min_preco )  #valor posicionado atual
            
        preco = self.df.loc[self.current_step, "preco"]
        preco_ant = self.df.loc[self.current_step - 1, "preco"]
        delta_preco = preco - preco_ant
        dp = ( preco * ( self.max_preco - self.min_preco ) + self.min_preco ) - valor_cheio #variao do preo atual e do preo de compra/venda
        posicao = self.ncont_anterior * dp * 10 - self.custo * abs(acao) #posicao = lucro - custo (INSTANTNEO)
    
        #calculos sobre o valor    
        if ( self.ncont_anterior == 0 and acao != 0 ): #primeiro valor
            self.valor = preco
        elif ( self.ncont == 0 ):
            self.valor = 0
        
        if(self.ncont != self.ncont_anterior):
            return posicao
        elif(self.ncont == 1):
            return delta_preco
        elif(self.ncont == 0):
            return -delta_preco

    def step(self, action):
        # Execute one time step within the environment
        reward = self._take_action(action)
        self.reward = reward
        self.sum_rewards += reward

        self.current_step += 1
        done = False
        if(self.current_step == self.df.shape[0] - 1):
            done = True
        elif (self.df.iloc[self.current_step]['dt'] != self.df.iloc[self.current_step + 1]['dt']):
            done = True

        obs = self._next_observation()
        
        return obs, reward, done, {}

    def reset(self):
        self.ncont = 0
        self.ncont_anterior = 0
        self.valor = 0
        self.reward = 0
        self.sum_rewards = 0
        self.current_step = self.comeco_dias[random.randrange(len(self.comeco_dias))] + self.passos - 1
        return self._next_observation()

    def render(self, mode='human', close=False):
        print('{0} {1}'.format(self.df.iloc[self.current_step]['dt'], tempoIntToStr(self.df.iloc[self.current_step]['hr_int'])))
        print('Preco: {0}'.format(self.df.iloc[self.current_step]['preco']))
        print('ncont: {0}'.format(self.ncont))
        print('reward: {0}'.format(self.reward))
        print('ganho: {0}'.format(self.sum_rewards))
        print('---------------------------')
