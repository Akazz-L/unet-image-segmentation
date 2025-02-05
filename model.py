from keras.models import *
from keras.layers import *
from keras.optimizers import *

# paper : https://arxiv.org/pdf/1505.04597v1.pdf
def unet_model(pretrained_weights = None, input_size=(512,512,1), dropout = False):
    
    inputs = Input(input_size)
    
    # Downscale
    
    conv1 = Conv2D(64,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) # By default, the paper does not use any padding 
    conv1 = Conv2D(64,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1)
    
    conv2 = Conv2D(128,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size= (2,2))(conv2)

    
    conv3 = Conv2D(256,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size = (2,2))(conv3)

    
    conv4 = Conv2D(512,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) # COPY&CROP
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size = (2,2))(conv4)

    if dropout:
        pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(1024,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024,3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    if dropout:
        conv5 = Dropout(0.5)(conv5)

    # Upscale
    upscale6 = Conv2D(512,3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D( size = (2,2))(conv5))
    merge6 = concatenate([conv4,upscale6], axis = 3)
    conv6 = Conv2D(512,3,activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512,3,activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #conv6 = BatchNormalization()(conv6)
    
    upscale7 = Conv2D(256,3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D( size = (2,2))(conv6))
    merge7 = concatenate([conv3,upscale7], axis = 3)
    conv7 = Conv2D(256,3,activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256,3,activation = 'relu',padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #conv7 = BatchNormalization()(conv7)


    upscale8 = Conv2D(128,3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D( size = (2,2))(conv7))
    merge8 = concatenate([conv2, upscale8], axis = 3)
    conv8 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #conv8 = BatchNormalization()(conv8)


    upscale9 = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D( size = (2,2))(conv8))
    merge9 = concatenate([conv1, upscale9], axis = 3)
    conv9 = Conv2D(64,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = BatchNormalization()(conv9)


    conv10 = Conv2D(1,1, activation = 'sigmoid')(conv9)
    

    model = Model(inputs=inputs, outputs=conv10, name = "unet_model")

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
                                                                                        
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

