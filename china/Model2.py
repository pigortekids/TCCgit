import tensorflow as tf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline

arquivo = pd.read_csv("H:/TCC/ArquivoFinal/v7/Consolidado.csv")
arquivo = arquivo[['preco', 'hr_int', 'preco_pon', 'qnt_soma', 'max', 'min']]
arquivo = arquivo.astype({'preco': float, 'hr_int': int, 'preco_pon': float,
                              'qnt_soma': int, 'max': float, 'max': float})

#TCS_df = pd.read_csv('TCS')
#p = TCS_df['Close'].as_matrix()
l = []

tf.reset_default_graph()

file_size = arquivo.shape[0]
variables = arquivo.shape[1] #numero de variaveis a se considerar na entrada
memory = 100 * variables #time steps
n_inputs = 10 * variables #no of price ticks[inputs]
trans_cost = 1 #transaction cost
batch_size = 1
n_layers = 5
n_nodes = 128
#features shape - (m*1)
#W - shape(m*1)
#U - shape(1*1)

data_mean = []
for i in range(arquivo.shape[1]):
    arquivo.iloc[:,i] = ( arquivo.iloc[:,i] - arquivo.iloc[:,i].min() ) / ( arquivo.iloc[:,i].max() - arquivo.iloc[:,i].min() )
inputs = []
print(-3)
for i in range(file_size):
    for j in range(variables):
        inputs.append(arquivo.iloc[i,j])
print(-2)
inputs = np.array(inputs)
inputs = inputs.reshape([file_size * variables,1])
print(-1)
decisions = np.array([1,0,-1])

def next_batch(t,test=False):
    l1 = ind
    l2 = ind+n_inputs+memory
    return inputs[l1:l2]

def init_weights(shape):
    rn = tf.random_normal(shape)
    return tf.Variable(rn)

def inp_layer(x,camada_entrada,b):
    return tf.matmul(tf.transpose(camada_entrada),x) + b

def out_inp_layer(d,U):
    return tf.matmul(tf.transpose(U),d)

def output(x,y):
    return tf.tanh(x+y)

def forward_prop(ft,dt,camada_entrada,b,U):
    i1 = inp_layer(ft,camada_entrada,b)
    i2 = out_inp_layer(dt,U)
    o = output(i1,i2)
    return o

def unfolded(features,camada_entrada,b,U,dt,Wdeep,bdeep):
    delta = [dt]
    r = []
    for i in range(1,memory+1):
        f = tf.slice(features,[i-1,0],[n_inputs,1])
        f = dnn_layer(f,Wdeep,bdeep)
        l.append(f)
        delta.append(forward_prop(f,delta[i-1],camada_entrada,b,U))
        temp = delta[i-1]*inputs[i-1] - trans_cost*tf.abs(delta[i] - delta[i-1])
        r.append(temp)
    UT = sum(r)
    return UT,r,delta

def dnn_layer(features,Wdeep,bdeep):
    inp = features
    for i in range(n_layers):
        inp = tf.matmul(tf.transpose(Wdeep[i]),inp) + bdeep[i]
        inp = tf.nn.sigmoid(inp)
    return inp

print(0)
plt.plot([i for i in range(n_inputs)],inputs[0:n_inputs],'r')
plt.plot([i for i in range(n_inputs,n_inputs+memory)],inputs[n_inputs:n_inputs+memory],'g')

learning_rate = 0.01

obj = []
print(1)
camada_entrada = init_weights([n_inputs,1])
U = init_weights([1,1])
b = init_weights([1,1])
print(2)
dt = tf.Variable(tf.zeros([1,1]),dtype=tf.float32,trainable=False)
features = tf.placeholder(tf.float32,shape = [n_inputs+memory,1])
print(3)
Wdeep = [init_weights([n_inputs,n_nodes])]
bdeep = [init_weights([n_nodes,1])]
print(4)
for i in range(n_layers-2):
    Wdeep.append(init_weights([n_nodes,n_nodes]))
    bdeep.append(init_weights([n_nodes,1]))
print(5)
bdeep.append(init_weights([n_inputs,1]))
Wdeep.append(init_weights([n_nodes,n_inputs]))
print(6)
UT,r,delta = unfolded(features,camada_entrada,b,U,dt,Wdeep,bdeep)
print(7)
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
print(8)
train = optimizer.minimize(-UT)
print(9)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
steps = 1
epochs = 10000
with tf.Session() as sess:
    sess.run(init)
    for j in range(epochs):
        ind = 0
        for i in range(steps):
            #print(ind)
            x_curr = next_batch(ind)
            #_,val,delt,d = sess.run([train,UT,delta,dt],feed_dict = {features:x_curr.reshape([10,1])})
            _,val,delt,w = sess.run([train,UT,delta,Wdeep],feed_dict = {features:x_curr.reshape([n_inputs+memory,1])})
            obj.append(val[0][0])
            ind = ind + n_inputs+memory
            print(val)
    saver.save(sess, "./stock")
    
plt.plot(obj,'r')