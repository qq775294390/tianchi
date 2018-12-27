import sys
reload(sys)
sys.setdefaultencoding('utf8')
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras




L_inputs=[None]*18
for i in range(18):
    L_inputs[i]=keras.layers.Input((32,32,1))
X=L_inputs[:]

conv0=keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1),name='conv0')
pool0=keras.layers.MaxPool2D((2, 2),name='pool0')
conv1=keras.layers.Conv2D(64, (3, 3), activation='relu',name='conv1')
pool1=keras.layers.MaxPool2D((2, 2),name='pool1')
conv2=keras.layers.Conv2D(32, (3, 3), activation='relu',name='conv2')
flat=keras.layers.Flatten(name='flat')
den0=keras.layers.Dense(64, activation='relu',name='den0')

for i in range(18):
    X[i] = conv0(X[i])
    X[i] = pool0(X[i])
    X[i] = conv1(X[i])
    X[i] = pool1(X[i])
    X[i] = conv2(X[i])
    X[i] = flat(X[i])
    X[i] = den0(X[i])
X=keras.layers.Concatenate()(X)
X=keras.layers.Reshape((18,-1))(X)
X=keras.layers.Conv1D(12,1)(X)
X=keras.layers.MaxPool1D()(X)
X=keras.layers.Flatten()(X)
X=keras.layers.Dense(17,activation='softmax')(X)

model=keras.models.Model(inputs=L_inputs,outputs=X)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

fid = h5py.File('validation.h5', 'r')
# Loading sentinel-1 data patches
s1 = np.array(fid['sen1'])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'])
# Loading labels
labels = np.array(fid['label'])
model.fit([s1[:20000,:,:,i].reshape((-1,32,32,1)) for i in range(8)]
          +[s2[:20000,:,:,i].reshape((-1,32,32,1)) for i in range(10)]
          ,labels[:20000],epochs=10,batch_size=500)
res=model.evaluate([s1[20000:,:,:,i].reshape((-1,32,32,1))for i in range(8)]
                   +[s2[20000:,:,:,i].reshape((-1,32,32,1)) for i in range(10)]
                   ,labels[20000:])



print res
exit()