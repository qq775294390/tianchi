import sys
reload(sys)
sys.setdefaultencoding('utf8')
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

res_size=[(64,64,256),(128,128,512),(256,256,1024),(512,512,2048)]
L=[3,4,6,2]

L_input1=keras.layers.Input((32,32,8))
L_input2=keras.layers.Input((32,32,10))

X=keras.layers.Concatenate()([L_input1,L_input2])


X=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(X)
X=keras.layers.BatchNormalization(axis=-1)(X)
X=keras.layers.MaxPool2D((2,2))(X)

for i in range(4):
    for j in range(L[i]):
        key=res_size[i]
        if j==0:
            X_short = keras.layers.Conv2D(key[2],(1,1),activation='relu',padding='same')(X)
            X_short = keras.layers.BatchNormalization()(X_short)
        else:
            X_short = X
        X = keras.layers.Conv2D(key[0],(1,1),activation='relu',padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Conv2D(key[1], (3, 3), activation='relu',padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Conv2D(key[2], (3, 3), activation='relu', padding='same')(X)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Add()([X,X_short])
        X = keras.layers.Activation('relu')(X)
X=keras.layers.AveragePooling2D((2,2))(X)
X=keras.layers.Flatten()(X)
L_out=keras.layers.Dense(17,activation='softmax')(X)

model=keras.models.Model(inputs=[L_input1,L_input2],outputs=L_out)
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

fid = h5py.File('training.h5', 'r')
# Loading sentinel-1 data patches
s1 = np.array(fid['sen1'])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'])
# Loading labels
labels = np.array(fid['label'])

for i in range(10):
    model.fit(x=[s1,s2],y=labels,epochs=1,batch_size=64)
    model.save('model'+str(i))

exit()