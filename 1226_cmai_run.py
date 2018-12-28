import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
fid_train = h5py.File('../training.h5', 'r')
fid_val = h5py.File('../validation.h5','r')
train_len = len(fid_train['sen1'])
step = 50000
epoch = 1
train_epoch = 10
res=[]
models=[]
m0s=[]
for i in range(18):
    print(str(i))
    L_in=keras.layers.Input((32,32,1),name=str(i)+'input')

    X = keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,1),name=str(i)+'con0')(L_in)
    X = keras.layers.MaxPool2D((2,2),name=str(i)+'poo0')(X)
    X = keras.layers.Conv2D(32,(3,3),activation='relu',name=str(i)+'con1')(X)
    X = keras.layers.MaxPool2D((2,2),name=str(i)+'poo1')(X)
    X = keras.layers.Conv2D(32,(3,3),activation='relu',name=str(i)+'con2')(X)
    X = keras.layers.Flatten(name=str(i)+'fla')(X)
    X = keras.layers.Dense(64,activation='relu',name=str(i)+'den0')(X)
    out1=X
    X = keras.layers.Dense(17,activation='softmax',name=str(i)+'den1')(X)

    model = keras.models.Model(inputs=L_in,outputs=X)
    model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

    m0 = keras.models.Model(inputs=L_in, outputs=out1)
    m0s.append(m0)
    models.append(model)
for now_epoch in range(epoch):
    for j in range(int(train_len/step)):
        start_point = j*step

        if j+1 == int(train_len/step):
            end_point = train_len
        else:
            end_point = (j+1)*step
        print(str(start_point), str(end_point))
        # Loading sentinel-1 data patches
        s1_train = np.array(fid_train['sen1'][start_point:end_point])
        # Loading sentinel-2 data patches
        s2_train = np.array(fid_train['sen2'][start_point:end_point])
        # Loading labels
        labels_train = np.array(fid_train['label'][start_point:end_point])
        if i<8:
            X=s1_train[:,:,:,i].reshape((-1,32,32,1))
        else:
            X=s2_train[:,:,:,i-8].reshape((-1,32,32,1))
        y = labels_train
        for model in models:
            model.fit(X,y,epochs=train_epoch,batch_size=64)
            s1_val= np.array(fid_val['sen1'])
            s2_val = np.array(fid_val['sen2'])
            labels_val = np.array(fid_val['label'])
            if i<8:
                X_val=s1_val[:,:,:,i].reshape((-1,32,32,1))
            else:
                X_val=s2_val[:,:,:,i-8].reshape((-1,32,32,1))
            y_val = labels_val
            print(model.evaluate(X_val,y_val))
i = 0
for model in models:
    model.save('model'+str(i))
    i+=1
    # m0.save('m'+str(i))

models=[]
for i in range(18):
    models.append(keras.models.load_model('model'+str(i)))

X=keras.layers.Concatenate()([models[i].output for i in range(18)])
print(X)
X=keras.layers.Reshape((18,17))(X)
print(X)
X=keras.layers.Conv1D(12,1)(X)
X=keras.layers.Flatten()(X)
X=keras.layers.Dense(17,activation='softmax')(X)

model=keras.models.Model(inputs=[models[i].input for i in range(18)],outputs=X)


model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
for now_epoch in range(epoch):
    for i in range(int(train_len/step)):
        start_point = j * step
        if j + 1 == int(train_len / step):
            end_point = train_len
        else:
            end_point = (j + 1) * step
        # Loading sentinel-1 data patches
        s1_train = np.array(fid_train['sen1'][start_point:end_point])
        # Loading sentinel-2 data patches
        s2_train = np.array(fid_train['sen2'][start_point:end_point])
        # Loading labels
        labels_train = np.array(fid_train['label'][start_point:end_point])
        model.fit([s1_train[:,:,:,i].reshape((-1,32,32,1)) for i in range(8)]
                  +[s2_train[:,:,:,i].reshape((-1,32,32,1)) for i in range(10)]
                  ,labels_train,epochs=train_epoch,batch_size=64)
    s1_val = np.array(fid_val['sen1'])
    s2_val = np.array(fid_val['sen2'])
    labels_val = np.array(fid_val['label'])
    res=model.evaluate([s1_val[:,:,:,i].reshape((-1,32,32,1))for i in range(8)]
                       +[s2_val[:,:,:,i].reshape((-1,32,32,1)) for i in range(10)]
                       ,labels_val)
    print(res)
model.save('result')