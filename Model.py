import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *  
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from  config import Constant
import numpy as np

class MyCustomLayer(Layer): 
    def __init__(self):
        super(MyCustomLayer, self).__init__()
        self.trainable = False 
    @tf.function   
    def call(self, X): 
            
            batchsize = X.get_shape().as_list()[0]
            if(Constant.num_points==136):
            # for 300w
                Landmark=tf.reshape(X,(-1,68,2))
            elif(Constant.num_points==38):
            #for AFLW 
                Landmark=tf.reshape(X,(-1,19,2))
            else:    
            #for WFLW 
                Landmark=tf.reshape(X,(-1,98,2)) 
                
            num_landmark =Landmark.get_shape().as_list()[1]
            W,H=128,128
            sigma = 5
            hm=[]
            for i in range(0,batchsize):
                heatmap=[]
                for j in range(0,num_landmark):
                    xL = Landmark[i,j,0]
                    yL=Landmark[i,j,1]
                    heatmap.append(self.gaussian_k(xL,yL,sigma,W, H))
                hm.append(tf.reduce_sum(heatmap,axis=0))       
            heatmaps= tf. convert_to_tensor(hm)
            heatmaps = tf.expand_dims(heatmaps,axis=3)
            return  heatmaps
    def gaussian_k(self,x0,y0,sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = tf.range(0, width, 1, float) 
        y = tf.range(0, height, 1, float)[:, tf.newaxis]
        return tf.math.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))        
    def get_config(self):
        config = super().get_config()
        return config        
            
        
        
def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def self_attention_block(tensor, nfilters, size=1, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Activation("Sigmoid")(x)
    return x

def create_Model(x, stage_num,num_keypoints = Constant.num_points, filters=64):
# down
    conv1 = conv_block(x, nfilters=filters)# nfilters=64
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)# nfilters=128
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)# nfilters=256
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)# nfilters=512
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_block(conv4_out, nfilters=filters*16)# nfilters=1024
    
    
# output layer For Coordinate 
    conv5_out = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = conv_block(conv5_out, nfilters=filters*8)# nfilters=512
    conv6_out = MaxPooling2D(pool_size=(2, 2))(conv6)
    conv7 = conv_block(conv6_out, nfilters=filters*4)# nfilters=256
    attention1 =self_attention_block(conv7,nfilters=filters*4)
    flt = Flatten()(conv7)
    dense1 = Dense(1024,activation='linear')(flt)
    coordinates_Output = Dense(num_keypoints,activation='linear',name="output_stage{}".format(stage_num))(dense1)
    return  coordinates_Output
    
    
def build_model(input_shape,pretrained_file=None):
    input_layer = Input(shape=input_shape,batch_size=Constant.batch_size ,name="Input_stage")
    output = []  
    #### stage1############
    out1=create_Model(input_layer,1)
    output.append(out1)
    
    # ###### lambada function############
    lambda_output=MyCustomLayer()(out1)
    
    # ##########concatenate#############3
    x = Concatenate()([input_layer, lambda_output])
    out2=create_Model(x,2)
    output.append(out2)
    ############ biuld model##############  
    opt = Adam (lr = Constant.LR)
    #create the  Model
    model = Model(inputs=input_layer, outputs= output, name='Mynet')
    model.summary()
    losses =['mae','mae'] 
    #compile the  Model
    model.compile(optimizer=opt ,loss=losses)
    if (pretrained_file):
        model.load_weights(pretrained_file)    
    return model