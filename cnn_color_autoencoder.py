from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.layers.core import Flatten

import matplotlib as m; m.use('Agg')
import matplotlib.pyplot as plt
import sys

""" color対応 """
from PIL import Image
import glob
import numpy as np
BY = 2
x_train = []
x_test  = []
for eg, name in enumerate(glob.glob("datasets/minimize/*")):
  B  = 10
  im = Image.open(name)
  arr = np.array(im)
  if eg < 500*B:
    x_train.append( arr )
  if eg < 800*B:
    x_test.append( arr )
  if eg > 1000*B:
    break
  print(eg, arr.shape)

x_train = np.array(x_train)
x_test  = np.array(x_test)

#(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ...
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28*BY, 28*BY, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("orig.png")

input_img = Input(shape=(28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
encoder = Model(input_img, encoded)
# at this point the representation is (7, 7, 32)

""" dec network """
dec_1 = Conv2D(32, (3, 3), activation='relu', padding='same')
dec_2 = UpSampling2D((2, 2))
dec_3 = Conv2D(32, (3, 3), activation='relu', padding='same')
dec_4 = UpSampling2D((2, 2))
dec_5 = Conv2D(3, (3, 3), activation='relu', padding='same')
#dec_6 = Flatten()
#dec_7 = Dense(784)
""" define tensorflow """
x     = dec_1(encoded)
x     = dec_2(x)
x     = dec_3(x)
x     = dec_4(x)
x     = dec_5(x)
#x     = dec_6(x)
#x     = dec_7(x)
autoencoder = Model(input_img, x)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

""" build decoder """
print(encoded.shape)
enc_in = Input(shape=(7*BY,7*BY,32,))
x     = dec_1(enc_in)
x     = dec_2(x)
x     = dec_3(x)
x     = dec_4(x)
x     = dec_5(x)
decoder = Model(enc_in, x)

"""
autoencoder.fit(x_train_noisy, x_train,
  epochs=100,
  batch_size=128,
  shuffle=True,
  validation_data=(x_test_noisy, x_test),
	callbacks=[ log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
"""
if '--train' in sys.argv:
  autoencoder.fit(x_train_noisy, x_train,
    epochs=100,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    )
  autoencoder.save('models/cnn_model.h5')
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
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
