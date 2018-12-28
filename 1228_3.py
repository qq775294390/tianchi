
from tensorflow import keras
import numpy as np
import pandas as pd
import h5py

import gc
import h5py
model=keras.models.load_model('models/1227/resnet-6ep')
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['acc'])

fid = h5py.File('validation.h5', 'r')
# Loading sentinel-1 data patches
s1 = np.array(fid['sen1'])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'])
# Loading labels
labels = np.array(fid['label'])

model.fit([s1[:20000],s2[:20000]],labels[:20000])

print(model.evaluate([s1[:20000],s2[:20000]],labels[:20000]))

print( model.evaluate([s1[20000:],s2[20000:]],labels[20000:]) )

exit()