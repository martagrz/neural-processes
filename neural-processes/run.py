import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import collections
from model import NPModel

def get_y(x):
    y = np.sin(2*x) + 3*np.cos(x/2) + np.random.normal(loc=0, scale=0.01)
    return y

N = 500 
x_context = np.random.uniform(0,1,N)
y_context = get_y(x_context)
y_context = np.reshape(y_context, (N,1))
x_context = np.reshape(x_context, (N,1))

x_target = np.linspace(0,1,N)
y_target = get_y(x_target)
y_target = np.reshape(y_target, (N,1))
x_target = np.reshape(x_target, (N,1))
print('y shape', y_target.shape)

model = NPModel(x_context,y_context,x_target)

optimizer = tf.keras.optimizers.Adam(1e-2)

model.compile(optimizer=optimizer,loss = model.loss,metrics=[model.rms])

model.fit(x_target,y_target, batch_size=7, epochs=1000)


print('Done')                                    
        
