import sys
reload(sys)
sys.setdefaultencoding('utf8')
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
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
for i in range(18):
    print ''
    print i

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
    if i<8:
        X=s1[:20000,:,:,i].reshape((-1,32,32,1))
    else:
        X=s2[:20000,:,:,i-8].reshape((-1,32,32,1))
    y = labels[:20000]

    model.fit(X,y,epochs=10)

    if i<8:
        X=s1[20000:,:,:,i].reshape((-1,32,32,1))
    else:
        X=s2[20000:,:,:,i-8].reshape((-1,32,32,1))
    y = labels[20000:]
    res.append(model.evaluate(X,y))
    models.append(model)
    model.save('model'+str(i))
    m0.save('m'+str(i))

print('test')
exit()