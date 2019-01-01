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

model=keras.models.load_model('')
model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
fid_train = h5py.File('../training.h5', 'r')
fid_val = h5py.File('../validation.h5','r')
step=50000
length = len(fid_train['sen1'])
res = []
is_val=True
s1,s2,labels=None,None,None
if is_val:
    s1_val = fid_val['sen1']
    s2_val = fid_val['sen2']
    labels_val = fid_val['label']


for j in range(10):

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
            temp= model.evaluate(x=[s1_val[20000:],s2_val[20000:]],y=labels_val[20000:])
            print(model.evaluate(x=[s1_val[:20000], s2_val[:20000]], y=labels_val[:20000]))
            print(temp)
            print('*********************************')
            model.save(str((j,i))+str(temp[1]) + '-epoch')
