import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
import gc
is_val = True
res_size=[(64,64,256),(128,128,512),(256,256,1024),(512,512,2048)]
L=[3,4,6,2]

L_input1=keras.layers.Input((32,32,8))
L_input2=keras.layers.Input((32,32,10))

X=keras.layers.Concatenate()([L_input1,L_input2])


X=keras.layers.Conv2D(64,(3,3),activation='relu',padding='same')(X)
X=keras.layers.BatchNormalization(axis=-1)(X)
X=keras.layers.MaxPool2D((2,2))(X)

models=[]


for i in range(4):

    print(i)

    for j in range(L[i]):

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


        out = keras.layers.AveragePooling2D((2, 2))(X)
        out = keras.layers.Flatten()(out)
        out = keras.layers.Dense(64,activation='relu')(out)
        out = keras.layers.Dense(17,activation='softmax')(out)
        model = keras.models.Model(inputs=[L_input1, L_input2], outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        models.append(model)

X=keras.layers.AveragePooling2D((2,2))(X)
X=keras.layers.Flatten()(X)
L_out=keras.layers.Dense(17,activation='softmax')(X)

model=keras.models.Model(inputs=[L_input1,L_input2],outputs=L_out)
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

fid_train = h5py.File('../training.h5', 'r')
fid_val = h5py.File('../validation.h5','r')
step=50000
length = len(fid_train['sen1'])
res = []
s1,s2,labels=None,None,None
if is_val:
    s1_val = fid_val['sen1']
    s2_val = fid_val['sen2']
    labels_val = fid_val['label']
layers = 0
for mm in models:

    for i in range(int(length / step)):
        # Loading sentinel-1 data patches
        start_point = i*step
        if i+1 == int(length/step):
            end_point = length
        else:
            end_point = (i+1) * step
        del s1,s2,labels
        gc.collect()
        print ('loading')
        s1 = np.array(fid_train['sen1'][start_point:end_point])
        # Loading sentinel-2 data patches
        s2 = np.array(fid_train['sen2'][start_point:end_point])
        # Loading labels
        labels = np.array(fid_train['label'][start_point:end_point])

        mm.fit(x=[s1[:45000],s2[:45000]],y=labels[:45000],epochs=1,batch_size=64)
        if is_val:
            temp = mm.evaluate(x=[s1[45000:], s2[45000:]], y=labels[45000:])
            temp0 = mm.evaluate(x=[s1[:45000], s2[:45000]], y=labels[:45000])
            print('')
            print('---------------------------------------')
            print(temp)
            print(temp0)
    if is_val:
        mm.fit(x=[s1_val[:20000],s2_val[:20000]],y=labels_val[:20000],epochs=1,batch_size=64)
        temp=mm.evaluate(x=[s1_val[20000:],s2_val[20000:]],y=labels_val[20000:])
        print('')
        print('---------------------------------------')
        print(temp)
        res.append( temp )
    mm.save(str(layers)+'-models')
acc=[]
for i in range(10):

    for i in range(int(length / step)):
        start_point = i*step
        if i+1 == int(length/step):
            end_point = length
        else:
            end_point = (i+1) * step
        # Loading sentinel-1 data patches
        s1 = np.array(fid_train['sen1'][start_point:end_point])
        # Loading sentinel-2 data patches
        s2 = np.array(fid_train['sen2'][start_point:end_point])
        # Loading labels
        labels = np.array(fid_train['label'][start_point:end_point])

        model.fit(x=[s1,s2],y=labels,epochs=1,batch_size=64)
        if is_val:
            model.fit(x=[s1_val[:20000], s2_val[:20000]], y=labels_val[:20000], epochs=1, batch_size=64)
            print('*********************************')
            print(model.evaluate(x=[s1_val[20000:],s2_val[20000:]],y=labels_val[20000:]))
    if is_val:
        acc.append( model.evaluate(x=[s1_val,s2_val],y=labels_val) )
    model.save(str(i)+'-epoch')
exit()