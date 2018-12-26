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
for i in range(18):
    models.append(keras.models.load_model('model'+str(i)))

X=keras.layers.Concatenate()([models[i].output for i in range(18)])
X=keras.layers.Reshape((18,-1))(X)
X=keras.layers.Conv1D(12,1)(X)
X=keras.layers.Flatten()(X)
X=keras.layers.Dense(17,activation='softmax')(X)

model=keras.models.Model(inputs=[models[i].input for i in range(18)],outputs=X)

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
print res
exit()