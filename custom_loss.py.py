import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, models, optimizers
import tensorflow as tf
import math



#constantes
hbar = 1
omega = 1
m = 1



#bornes de discrétisation
a = -5
b = 5
#discrétisation de l'espace sur 1000 points
pts=1000
#array selon x
linx = np.linspace (a,b,pts)
#Position initiale
x=-5.
#pas d'incrémentation pour le calcul de la fonction d'onde
h = 10/pts
#calcul de la fonction d'onde
wave = np.zeros_like(linx)
for i in range(0,pts):
  wave[i] = pow(m*omega/(math.pi*hbar),0.25)*math.exp(-m*omega*(pow(x,2))/(2*hbar))
  x+=h












########
# SECTION : Approximation par machine learning
########







#custom loss
def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)




#Approximation par machine learning
model = models.Sequential([
    layers.Dense(200, input_shape=(1,), activation='relu'),
    layers.Dense(200, input_shape=(1,), activation='relu'),    
    layers.Dense(1), # no activation -> linear function of the input
])
model.summary()
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss=my_loss_fn,optimizer=opt)

model.fit(linx,wave,batch_size=50,epochs=45)
predictions = model.predict(linx)
preds = predictions.reshape(-1)




plt.title('Approx. par N.N. avec custom loss')
plt.plot(linx,wave,c='r',label = 'true wave function')
plt.plot(linx[0:pts-1:10],preds[0:pts-1:10],marker='x',c='forestgreen',label = 'custom loss',linestyle='None')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.savefig('custom_loss.pdf')