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

""" モデルを読み込み """
from cnn_deep_color_model import *

BY = 4
WIDTH = 2
x_train = []
x_test  = []
for eg, name in enumerate(glob.glob("../minimize/*")):
  B  = 230
  try:
    im = Image.open(name)
  except OSError as e:
    print(e)
    continue
  arr = np.array(im)
  try:
    arr = arr.astype('float32') / 255.
  except TypeError as e:
    print(e)
    continue
  if eg < 950*B:
    x_train.append( arr )
  else:
    x_test.append( arr )
  if eg > 1000*B:
    break
  if eg % 100 == 0:
    print(eg, arr.shape)

x_train = np.array(x_train)
x_test  = np.array(x_test)

x_train = np.reshape(x_train, (len(x_train), 28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28*BY, 28*BY, 3))  # adapt this if using `channels_first` image data format

if '--train' in sys.argv:
  for i in range(100):
    autoencoder.fit(x_train, x_train, \
      epochs=1, \
      batch_size=128, \
      shuffle=True, \
      validation_data=(x_test, x_test) )
    autoencoder.save('models/cnn_model_%08d.h5'%i)

if '--resume-train' in sys.argv:
  target = sorted(glob.glob("models/cnn_model_*.h5")).pop()
  autoencoder.load_weights(target)
  for i in range(100):
    autoencoder.fit(x_train, x_train, \
      epochs=1, \
      batch_size=128, \
      shuffle=True, \
      validation_data=(x_test, x_test) )
    autoencoder.save('models/cnn_model_%08d.h5'%i)

if '--eval' in sys.argv: 
  target = sorted(glob.glob("models/cnn_model_*.h5")).pop()
  print(target)
  #sys.exit()
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
