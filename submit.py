import sys
reload(sys)
sys.setdefaultencoding('utf8')
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

fid=h5py.File('round1_test_a_20181109.h5')
s1 = np.array(fid['sen1'])
# Loading sentinel-2 data patches
s2 = np.array(fid['sen2'])
model=keras.models.load_model('total')


pre=model.predict([s1[:,:,:,i].reshape((-1,32,32,1))for i in range(8)]
                   +[s2[:,:,:,i].reshape((-1,32,32,1)) for i in range(10)]
                   )
np.savetxt('zz.txt',pre)
res=np.zeros(pre.shape,dtype=np.int)
for i in range(len(res)):
    res[i][pre[i].argmax()]=1
zz=pd.DataFrame(res)
zz.to_csv('kkkkkk.csv',header=0,index=0)

tt=0