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

L_input=keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(32,32,1),padding='same')
X=L_input
for i in range(4):

    key=32

    X_short = keras.layers.Conv2D(key,(1,1),activation='relu',padding='same')(X)
    X=keras.layers.Conv2D(key,(1,1),activation='relu',padding='same')(X)
    X = keras.layers.Conv2D(key*2, (3, 3), activation='relu',padding='same')(X)
    X = keras.layers.Conv2D(key, (3, 3), activation='relu', padding='same')(X)
    X = keras.layers.Add()([X,X_short])
    X = keras.layers.Activation('relu')(X)
X=keras.layers.Flatten()(X)
X=keras.layers.Dense(128,activation='relu')(X)
L_out=keras.layers.Dense(17,activation='softmax')(X)

model=keras.models.Model(inputs=L_input,outputs=L_out)
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

fid = h5py.File('validation.h5', 'r')
# Loading sentinel-1 data patches
s1 = np.array(fid['sen1'][:20000])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'])
# Loading labels
labels = np.array(fid['label'])


X=s1[:20000,:,:,0].reshape((-1,32,32,1))
y=labels[:20000]

model.fit(x=X,y=y,epochs=10,batch_size=1)
print model.evaluate(s1[20000:,:,:,0].reshape((-1,32,32,1)),labels[20000:])

exit()