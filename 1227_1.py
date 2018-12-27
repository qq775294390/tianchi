import sys
reload(sys)
sys.setdefaultencoding('utf8')
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

models=[]
A=[None]*18
B=[None]*18
for i in range(18):
    models.append(keras.models.load_model('models/1225/m'+str(i)))
    A[i]=models[i].output
    A[i]=keras.layers.Dense(32,activation='relu')(A[i])
    B[i]=keras.layers.Reshape((-1,1))(A[i])
    A[i]=keras.layers.Reshape((1,-1))(A[i])

X=keras.layers.Concatenate(axis=-2)(A)

X=keras.layers.Conv1D(12,1)(X)
X=keras.layers.Flatten()(X)


Y=keras.layers.Concatenate()(B)
Y=keras.layers.Conv1D(12,1)(Y)
Y=keras.layers.Flatten()(Y)
out=keras.layers.Concatenate()([X,Y])
out=keras.layers.Dense(17,activation='softmax')(out)

model=keras.models.Model(inputs=[models[i].input for i in range(18)],outputs=out)

fid = h5py.File('validation.h5', 'r')
# Loading sentinel-1 data patches
s1 = np.array(fid['sen1'])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'])
# Loading labels
labels = np.array(fid['label'])


model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
model.fit([s1[:20000,:,:,i].reshape((-1,32,32,1)) for i in range(8)]
          +[s2[:20000,:,:,i].reshape((-1,32,32,1)) for i in range(10)]
          ,labels[:20000],epochs=10)
res=model.evaluate([s1[20000:,:,:,i].reshape((-1,32,32,1))for i in range(8)]
                   +[s2[20000:,:,:,i].reshape((-1,32,32,1)) for i in range(10)]
                   ,labels[20000:])

fid = h5py.File('training.h5', 'r')
# Loading sentinel-1 data patches
s1 = np.array(fid['sen1'][10000:15000])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'][10000:15000])
# Loading labels
labels = np.array(fid['label'][10000:15000])

print model.evaluate([s1[:, :, :, i].reshape((-1, 32, 32, 1)) for i in range(8)]
                     + [s2[:, :, :, i].reshape((-1, 32, 32, 1)) for i in range(10)]

                     , labels[:])

print res
exit()