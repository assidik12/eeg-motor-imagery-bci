import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten, Dropout,
                                     BatchNormalization, Activation, AveragePooling2D, 
                                     DepthwiseConv2D, SeparableConv2D)
from tensorflow.keras.constraints import max_norm

def EEGNet(nb_classes, Chans=64, Samples=128, 
           dropoutRate=0.5, kernLength=64, F1=8, 
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """
    Implementasi Keras Functional API dari EEGNet.
    
    Arsitektur ini didasarkan pada notebook penelitian Anda dan artikel aslinya.
    URL: https://arxiv.org/abs/1611.08024
    
    Penting: Fungsi ini HANYA mengembalikan arsitektur model (belum di-compile).
    Proses kompilasi (menentukan optimizer, loss) akan dilakukan di 'train.py'.
    """
    
    # Pastikan kita menggunakan tipe Dropout yang valid
    if dropoutType == 'SpatialDropout2D':
        from tensorflow.keras.layers import SpatialDropout2D
        dropoutLayer = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutLayer = Dropout
    else:
        raise ValueError(f"dropoutType '{dropoutType}' tidak dikenali. "
                         "Gunakan 'SpatialDropout2D' atau 'Dropout'.")

    # Input Layer
    # Format Keras: (batch_size, height, width, channels)
    # Untuk EEG: (batch_size, num_channels, num_samples, 1)
    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    # Block 1: Temporal Convolution + Spatial (Depthwise) Convolution
    ##################################################################
    
    # Temporal Convolution
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    
    # Spatial (Depthwise) Convolution
    # Mempelajari filter spasial untuk setiap feature map dari temporal conv
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, 
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutLayer(dropoutRate)(block1)

    ##################################################################
    # Block 2: Separable Convolution
    # Menggabungkan feature map dan melakukan konvolusi temporal lagi
    ##################################################################
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutLayer(dropoutRate)(block2)

    ##################################################################
    # Classification Block
    ##################################################################
    flatten = Flatten(name='flatten')(block2)
    
    dense = Dense(nb_classes, name='dense', 
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

if __name__ == '__main__':
    # Bagian ini hanya untuk testing cepat
    # Jalankan file ini langsung (python src/model.py) untuk melihat summary
    
    print("Membuat model EEGNet untuk testing...")
    model = EEGNet(nb_classes=4, Chans=22, Samples=1000)
    model.summary()
    print("\nFile model.py berhasil di-load.")
