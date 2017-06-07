from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.layers.core import Flatten
from keras.layers import LeakyReLU, Activation, Lambda, Layer
from keras.layers.core import Reshape, Dropout
from keras.layers.normalization import BatchNormalization as BN
import matplotlib as m; m.use('Agg')
import matplotlib.pyplot as plt
import sys

""" color対応 """
from PIL import Image
import glob
import numpy as np

def sampling(args):
  epsilon_std = 1.0
  batch_size  = 128
  latent_dim  = 7*7*4
  z_mean, z_log_var = args
  epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
  return z_mean + K.exp(z_log_var / 2) * epsilon


BY = 4

input_img = Input(shape=(28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#x = Dropout(0.25)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = BN()(x)
x = Conv2D(256, (3, 3), activation='relu' , padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu' , padding='same')(x)
#x = Conv2D(256, (3, 3), activation='relu' , padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = BN()(x)
x = Conv2D(512, (3, 3), activation='relu' , padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu' , padding='same')(x)
#x = Conv2D(512, (3, 3), activation='relu' , padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = BN()(x)
x = Flatten()(x)
#encoded = Dense(392, activation='relu')(x)
z_mean    = Dense(196)(x)
#z_log_var = Dense(196)(x)
#z         = Lambda(sampling, output_shape=(196,))([z_mean, z_log_var])

encoder = Model(input_img, z_mean)
# at this point the representation is (7, 7, 32)

""" dec network """
dec_0  = Reshape((7,7,4))
dec_1  = Conv2D(32, (3, 3), padding='same')
dec_2  = LeakyReLU(0.2, name="leaky_d1")
dec_3  = UpSampling2D((2, 2))
dec_4  = Conv2D(64, (3, 3), padding='same')
dec_5  = LeakyReLU(0.2)
dec_6  = UpSampling2D((2, 2))
dec_7  = Conv2D(128, (3, 3), padding='same')
dec_8  = LeakyReLU(0.2)
dec_9  = UpSampling2D((2, 2))
dec_10 = Conv2D(128, (2, 2), padding='same')
dec_11 = LeakyReLU(0.2, name="leaky_d4")
dec_12 = UpSampling2D((1, 1))
dec_13 = Conv2D(128, (2, 2), padding='same')
dec_14 = LeakyReLU(0.2, name="leaky_d5")
dec_15 = UpSampling2D((2, 2))
dec_16 = Conv2D(3, (2, 2), padding='same')
dec_17 = LeakyReLU(0.2, name="leaky_d6")
""" define tensorflow """
x     = dec_0(z_mean)
x     = dec_1(x)
x     = BN()(dec_2(x))
x     = dec_3(x)
x     = dec_4(x)
x     = BN()(dec_5(x))
x     = dec_6(x)
x     = dec_7(x)
x     = BN()(dec_8(x))
x     = dec_9(x)
x     = dec_10(x)
x     = BN()(dec_11(x))
x     = dec_12(x)
x     = dec_13(x)
x     = BN()(dec_14(x))
x     = dec_15(x)
x     = dec_16(x)
x     = dec_17(x)
autoencoder = Model(input_img, x)
autoencoder.compile(optimizer='adam', loss='mse')

""" build decoder """
print(z_mean.shape)
# ここのサイズはアドホック
#enc_in = Input(shape=(392,))
enc_in = Input(shape=(196,))
x     = dec_0(enc_in)
x     = dec_1(x)
x     = BN()(dec_2(x))
x     = dec_3(x)
x     = dec_4(x)
x     = BN()(dec_5(x))
x     = dec_6(x)
x     = dec_7(x)
x     = BN()(dec_8(x))
x     = dec_9(x)
x     = dec_10(x)
x     = BN()(dec_11(x))
x     = dec_12(x)
x     = dec_13(x)
x     = BN()(dec_14(x))
x     = dec_15(x)
x     = dec_16(x)
x     = dec_17(x)
decoder = Model(enc_in, x)
