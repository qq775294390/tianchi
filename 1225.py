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

X=s1[:20000]
y=labels[:20000]

res=[]
models=[]
for i in range(8):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,1)))
    model.add(keras.layers.MaxPool2D((2,2)))
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
    model.add(keras.layers.MaxPool2D((2,2)))
    model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dense(17,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
    model.fit(X[:,:,:,i].reshape((-1,32,32,1)),y)
    res.append(model.evaluate(s1[20000:,:,:,i].reshape((-1,32,32,1)),labels[20000:]))
    models.append(model)


exit()