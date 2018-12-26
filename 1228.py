import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,\
    BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import pydoc
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format("channels_last")
K.set_learning_phase(1)
def identity_block(X,f,filters,stage,block):
    conv_name_base = "res"+str(stage)+block+"_branch"
    bn_name_base = "bn"+str(stage)+block+"_branch"
    F1,F2,F3 = filters
    X_shortcut = X

    X = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="same",activation='relu')(X)


    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",
               name=conv_name_base+"2b",kernel_initializer=glorot_uniform(seed=0),activation='relu')(X)

    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="same",
               name=conv_name_base+"2c",kernel_initializer=glorot_uniform(seed=0))(X)

    X = layers.add([X,X_shortcut])

    model=keras.models.Model(input=X_shortcut,outputs=X)
    return X
A_prev=Input((32,32,2))
A = identity_block(A_prev,f=2,filters=[2,4,6],stage=1,block="a")