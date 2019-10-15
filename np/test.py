import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd

def build_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

def get_y(x):
    y = np.sin(2*x) + 3*np.cos(x/2)
    return y

x = np.random.uniform(-10,10,100)
y = get_y(x)

x_test = np.linspace(-10,10,1000)
y_test = get_y(x_test)

print('Data generated')

plt.scatter(x,y,c='blue', label='Data')
plt.plot(x_test,y_test,c='black', label='Ground truth')
plt.legend(loc='upper right')
plt.show()

model = build_model()

EPOCHS = 10000

class PrintDot(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

history = model.fit(
    x, y, epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')


plot_history(history)


loss, mae, mse = model.evaluate(x_test, y_test, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
  
test_predictions = model.predict(x_test)

plt.figure()
plt.scatter(x,y,c='black',label='Data')
plt.plot(x_test,y_test,c='red',label='Ground truth')
plt.plot(x_test,test_predictions,c='blue',label='Predictions')
plt.legend(loc='upper right')
plt.show()





