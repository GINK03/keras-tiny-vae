from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.layers.core import Flatten
from keras.layers import LeakyReLU, Activation
from keras.layers.core import Reshape

import matplotlib as m; m.use('Agg')
import matplotlib.pyplot as plt
import sys
from keras.models import load_model

""" color対応 """
from PIL import Image
import glob
import numpy as np
BY = 4
x_train = []
x_test  = []
for eg, name in enumerate(glob.glob("minimize/*")):
  B  = 100
  im = Image.open(name)
  arr = np.array(im)
  try:
    arr = arr.astype('float32') / 255.
  except TypeError as e:
    print(e)
    continue
  if eg < 900*B:
    x_train.append( arr )
  else:
    x_test.append( arr )
  if eg > 1000*B:
    break
  if eg % 100 == 0:
    print(eg, arr.shape)

x_train = np.array(x_train)
x_test  = np.array(x_test)

#(x_train, _), (x_test, _) = mnist.load_data()

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format


input_img = Input(shape=(28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
#x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
x = Flatten()(x)
x = Dense(784, activation='linear')(x)
encoded = x
encoder = Model(input_img, encoded)
# at this point the representation is (7, 7, 32)

""" dec network """
dec_0  = Reshape((7,7,16))
dec_1  = Conv2D(32, (3, 3), padding='same')
dec_2  = LeakyReLU(0.2, name="leaky_d1")
dec_3  = UpSampling2D((2, 2))
dec_4  = Conv2D(32, (3, 3), padding='same')
dec_5  = LeakyReLU(0.2)
dec_6  = UpSampling2D((2, 2))
dec_7  = Conv2D(3, (3, 3), padding='same')
dec_8  = LeakyReLU(0.2)
dec_9  = UpSampling2D((2, 2))
dec_10 = Conv2D(3, (3, 3), padding='same')
dec_11 = LeakyReLU(0.2, name="leaky_d4")
dec_12 = UpSampling2D((2, 2))
dec_13 = Conv2D(3, (2, 2), padding='same')
dec_14 = LeakyReLU(0.2, name="leaky_d5")
#dec_15 = UpSampling2D((3, 3))
#dec_16 = Conv2D(3, (3, 3), padding='same')
#dec_17 = LeakyReLU(0.2, name="leaky_d6")
""" define tensorflow """
x     = dec_0(encoded)
x     = dec_1(x)
x     = dec_2(x)
x     = dec_3(x)
x     = dec_4(x)
x     = dec_5(x)
x     = dec_6(x)
x     = dec_7(x)
x     = dec_8(x)
x     = dec_9(x)
x     = dec_10(x)
x     = dec_11(x)
x     = dec_12(x)
x     = dec_13(x)
x     = dec_14(x)
#x     = dec_15(x)
#x     = dec_16(x)
#x     = dec_17(x)
autoencoder = Model(input_img, x)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

""" build decoder """
print(encoded.shape)
# ここのサイズはアドホック
enc_in = Input(shape=(784,))
x     = dec_0(enc_in)
x     = dec_1(x)
x     = dec_2(x)
x     = dec_3(x)
x     = dec_4(x)
x     = dec_5(x)
x     = dec_6(x)
x     = dec_7(x)
x     = dec_8(x)
x     = dec_9(x)
x     = dec_10(x)
x     = dec_11(x)
x     = dec_12(x)
x     = dec_13(x)
x     = dec_14(x)
#x     = dec_15(x)
#x     = dec_16(x)
#x     = dec_17(x)
decoder = Model(enc_in, x)

if '--train' in sys.argv:
  for i in range(50):
    autoencoder.fit(x_train, x_train, \
      epochs=1, \
      batch_size=128, \
      shuffle=True, \
      validation_data=(x_test, x_test) )
    autoencoder.save('models/cnn_model_%08d.h5'%i)

if '--eval' in sys.argv: 
  target = sorted(glob.glob("models/cnn_model_*.h5")).pop()
  print(target)
  autoencoder.load_weights(target)
n = 50  # how many digits we will display
plt.figure(figsize=(200, 40))
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28*BY, 28*BY, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    print(decoded_imgs[i].shape)
    plt.imshow(decoded_imgs[i].reshape(28*BY, 28*BY, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("denoise.png")
