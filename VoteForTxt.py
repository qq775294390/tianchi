import numpy as np

A=[None]*3
A[0]=np.loadtxt('1.txt')
A[1]=np.loadtxt('2.txt')

pre=A[0]+A[1]

res=np.zeros(pre.shape,dtype=np.int)
for i in range(len(res)):
    res[i][pre[i].argmax()]=1
np.savetxt('submit_zzz',res,fmt='%d',delimiter=',')