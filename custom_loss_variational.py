import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf
import math

from scipy import integrate
from scipy import interpolate
from scipy.stats import norm

import time

time1 = time.clock()

#constantes
hbar = 1
omega = 1
m = 1

fits = 100


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
wav2 = np.zeros_like(linx)
for i in range(0,pts):
  wave[i] = pow(m*omega/(math.pi*hbar),0.25)*math.exp(-m*omega*(pow(x,2))/(2*hbar))
  wav2[i] = pow(m*omega/(math.pi*hbar),0.25)*math.exp(-m*omega*(pow(0.9*x,2))/(2*hbar))
  x+=h





def compute_energy (abscissa,function):
  #interpolations
  tck_true = interpolate.splrep(abscissa, function, k=3, s=0)                                    #W.F.
  tck_true_carre = interpolate.splrep(abscissa, function*function, k=3, s=0)                     #W.F. squared
  tck_true_x = interpolate.splrep(abscissa, abscissa*abscissa*function*function, k=3, s=0)       #W.F. squared*x^2
  der_true = interpolate.splev(abscissa, tck_true, der=1)                                        #W.F. derivative
  tck_true_der = interpolate.splrep(abscissa,der_true*der_true, k=3,s=0)                         #W.F. derivative spline 1000
  int_true_carre = interpolate.splint(a,b,tck_true_carre)                                        #integral of W.F. squared
  int_true_x = interpolate.splint(a,b,tck_true_x)                                                #integral of W.F. squared*x^2 (<x^2>)
  int_true_der = interpolate.splint(a,b,tck_true_der)                                            #integral of derivative squared
  #energy
  Energy = ((-pow(hbar,2)/(2*m))*(function[-1]*der_true[-1]-function[0]*der_true[0] 
                             - int_true_der) + 0.5*m*omega*int_true_x ) / int_true_carre
  return Energy






########
# SECTION : Approximation par machine learning
########







#custom loss
def my_loss_fn(y_true, y_pred):
#    print('y_true:',y_true)
#    print('--')
#    print('y_pred:',y_pred)
    y_true = K.print_tensor(y_true, message='y_true = ')
    y_pred = K.print_tensor(y_pred, message='y_pred = ')
#    exit()
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference, axis=-1)
    mse = K.print_tensor(mse, message='mse = ')
    return mse

#custom loss
def my_loss_fn_variational(y_true, y_pred):
#    linx = np.linspace (-5,5,1000)
#    energy = compute_energy(linx,y_pred)
#    return energy
    squared_difference = tf.square(y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)



#Approximation par machine learning
model = models.Sequential([
    layers.Dense(20, input_shape=(1,), activation='relu'),
    layers.Dense(20, input_shape=(1,), activation='relu'),    
    layers.Dense(1), # no activation -> linear function of the input
])
model.summary()
opt = optimizers.Adam(learning_rate=0.001)
#
# use mse for the loss function:
#
#model.compile(loss='mse',optimizer=opt)
#
# use my own loss function defined as my_loss_fn:
#
model.compile(loss=my_loss_fn,optimizer=opt)
#
# use my own loss function defined as my_loss_fn_variational:
#
#model.compile(loss=my_loss_fn_variational,optimizer=opt)

for i in range(0,fits):
    model.fit(linx,wave,batch_size=50,epochs=1,verbose=0)
    predictions = model.predict(linx)
    # make the prediction positive
    preds = np.abs(predictions.reshape(-1))
    # make the prediction symmetric
    for j in range(0,pts):
        preds[j] = (preds[j]+preds[pts-1-j])/2
        preds[pts-1-j] = preds[j]
    # normalisation:
    # W.F. squared:
    tck_true_carre = interpolate.splrep(linx, preds*preds, s=0)
    #integral of W.F. squared
    int_true_carre = interpolate.splint(a,b,tck_true_carre)
    #new normalized function
    preds = preds/pow(int_true_carre,1/2)
    #
    # Compute energy
    energy_preds = compute_energy(linx,preds)
    print('Energy: ',energy_preds)

print('CPU time = ',time.clock()-time1)
    
plt.title('Approx. par N.N. avec custom loss')
plt.plot(linx,wave,c='r',label = 'true wave function')
plt.plot(linx[0:pts-1:10],preds[0:pts-1:10],marker='x',c='forestgreen',label = 'custom loss',linestyle='None')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.savefig('custom_loss_variational.pdf')
