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
import json
import re
import os.path

""" color対応 """
from PIL import Image
import glob
import numpy as np

""" モデルを読み込み """
from cnn_deep_color_model import *
target = sorted(glob.glob("models/cnn_model_*.h5")).pop()
print(target)
autoencoder.load_weights(target)

xs = []
ns = []
for eg, name in enumerate(glob.glob("../minimize/*")):
  n  =  re.search(r"mini\.(illust_id=\d{1,})\.png", name).group(1)
  if os.path.exists("vectors/{}.json".format(n)):
    continue
  if xs != [] and eg % 1000 == 0:
    print("try to predict, now batch is %d"%eg)
    vs = encoder.predict( np.array(xs) )
    for n, v in zip(ns, vs.tolist() ):
      print(n, len(v))
      open("vectors/{}.json".format(n), "w").write( json.dumps(v) )
    xs = []
    ns = []

  
  try:
    im = Image.open(name)
  except OSError as e:
    print(e)
    continue
  x = np.array(im)
  try:
    x = x.astype('float32') / 255.
  except TypeError as e:
    print(e)
    continue
  xs.append( x )
  ns.append( n )


