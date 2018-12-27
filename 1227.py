import sys
reload(sys)
sys.setdefaultencoding('utf8')
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
K = keras.backend

fid = h5py.File('validation.h5', 'r')
# Loading sentinel-1 data patches
s1 = np.array(fid['sen1'])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'])
# Loading labels
labels = np.array(fid['label'])

X_in=s1[:20000]
y=labels[:20000]

res=[]
models=[]
L_in=[None]*18
X=[None]*18
hiden=[None]*18

#public_D=keras.layers.Dense(64,activation='relu',name='den_public')
public_Out=keras.layers.Dense(17,activation='relu',name='den_out')

for i in range(18):
    print ''
    print i

    L_in[i]=keras.layers.Input((32,32,1),name=str(i)+'input')

    X[i] = keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,1),name=str(i)+'con0')(L_in[i])
    X[i] = keras.layers.MaxPool2D((2,2),name=str(i)+'poo0')(X[i])
    X[i] = keras.layers.Conv2D(32,(3,3),activation='relu',name=str(i)+'con1')(X[i])
    X[i] = keras.layers.MaxPool2D((2,2),name=str(i)+'poo1')(X[i])
    X[i] = keras.layers.Conv2D(32,(3,3),activation='relu',name=str(i)+'con2')(X[i])
    X[i] = keras.layers.Flatten(name=str(i)+'fla')(X[i])
    X[i] = keras.layers.Dense(64,activation='relu',name=str(i)+'den')(X[i])
    hiden[i]=X[i]
    X[i] = public_Out(X[i])

A=keras.layers.Concatenate()(hiden)
A=keras.layers.Reshape((18,-1))(A)
A=keras.layers.Conv1D(16,1)(A)
A=keras.layers.Flatten()(A)
A=keras.layers.Dense(17,activation='relu')(A)
MODEL=keras.models.Model(inputs=L_in,outputs=A)

B=[None]*18
for i in range(18):
    B[i]=keras.layers.Reshape((-1,1))(hiden[i])

A=keras.layers.Concatenate(axis=-1)(B)
A=keras.layers.Conv1D(16,1)(A)
A=keras.layers.Flatten()(A)
A=keras.layers.Dense(17,activation='relu')(A)
MODEL0=keras.models.Model(inputs=L_in,outputs=A)

for j in range(10):
    print j
    for i in range(18):
        print i

        model = keras.models.Model(inputs=L_in[i],outputs=X[i])
        model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
        if i<8:
            XX=s1[:20000,:,:,i].reshape((-1,32,32,1))
        else:
            XX=s2[:20000,:,:,i-8].reshape((-1,32,32,1))
        y = labels[:20000]

        model.fit(XX,y,epochs=1)

        if j==9:
            if i<8:
                XX=s1[20000:,:,:,i].reshape((-1,32,32,1))
            else:
                XX=s2[20000:,:,:,i-8].reshape((-1,32,32,1))
            y = labels[20000:]
            res.append(model.evaluate(XX,y))
            models.append(model)
            model.save('model'+str(i))
MODEL.save('MODEL_1227')
MODEL0.save('MODEL0_1227')

print('test')
exit()