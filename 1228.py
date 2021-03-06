import sys
reload(sys)
sys.setdefaultencoding('utf8')
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
import gc

res_size=[(64,64,256),(128,128,512),(256,256,1024),(512,512,2048)]
L=[3,4,6,2]

L_input1=keras.layers.Input((32,32,8))
L_input2=keras.layers.Input((32,32,10))

X=keras.layers.Concatenate()([L_input1,L_input2])


X=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(X)
X=keras.layers.BatchNormalization(axis=-1)(X)
X=keras.layers.MaxPool2D((2,2))(X)

models=[]


for i in range(1):

    print(i)

    for j in range(1):

        print((i,j))

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


        out = keras.layers.MaxPooling2D((2, 2))(X)
        out = keras.layers.Conv2D(12,1,activation='relu')(out)
        out = keras.layers.Flatten()(out)
        out = keras.layers.Dense(17,activation='softmax')(out)
        model = keras.models.Model(inputs=[L_input1, L_input2], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
        models.append(model)

# X=keras.layers.AveragePooling2D((2,2))(X)
# X=keras.layers.Flatten()(X)
# L_out=keras.layers.Dense(17,activation='softmax')(X)
#
# model=keras.models.Model(inputs=[L_input1,L_input2],outputs=L_out)
# model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['acc'])

fid = h5py.File('validation.h5', 'r')


length = len(fid['sen1'])
step=length

res = []
s1,s2,labels=None,None,None
for mm in models:

    for i in range(length / step):
        # Loading sentinel-1 data patches

        del s1,s2,labels
        gc.collect()
        print ('loading')
        s1 = np.array(fid['sen1'][0 * i:0 * i + step])
        # Loading sentinel-2 data patches
        s2 = np.array(fid['sen2'][0 * i:0 * i + step])
        # Loading labels
        labels = np.array(fid['label'][0 * i:0 * i + step])



        mm.fit(x=[s1[:20000],s2[:20000]],y=labels[:20000],epochs=1,batch_size=64,shuffle=False)
        temp=mm.evaluate(x=[s1[20000:],s2[20000:]],y=labels[20000:])

        print('')
        print('---------------------------------------')
        print(temp)
        res.append( temp )

acc=[]
for i in range(10):

    for i in range(length / step):
        # Loading sentinel-1 data patches
        s1 = np.array(fid['sen1'][0 * i:0 * i + step])
        # Loading sentinel-2 data patches
        s2 = np.array(fid['sen2'][0 * i:0 * i + step])
        # Loading labels
        labels = np.array(fid['label'][0 * i:0 * i + step])

        model.fit(x=[s1,s2],y=labels,epochs=1,batch_size=64)
        acc.append( model.evaluate(x=[s1[:20000],s2[:20000]],y=labels[:20000]) )
exit()