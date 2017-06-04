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
import plyvel
import pickle
""" color対応 """
from PIL import Image
import glob
import numpy as np

""" clustering """
from sklearn.cluster import KMeans

""" モデルを読み込み """
from cnn_deep_color_model import *

def step1():
  target = sorted(glob.glob("models/cnn_model_*.h5")).pop()
  print(target)
  autoencoder.load_weights(target)

  xs = []
  ns = []
  for eg, name in enumerate(glob.glob("../PixivImageScraper/minimize/*")):
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

""" ダウンロードデータと、ベクトルデータのファイルの積集合を取って、保存する """
def step2():
  print("step2")
  db = plyvel.DB('userid_metavec.ldb', create_if_missing=True)
  metas = set( n.split("/").pop() for n in glob.glob("../PixivImageScraper/metas/*") )
  vecs  = set( n.split("/").pop() for n in glob.glob("./vectors/*") )
  merge = metas & vecs
  for e, m in enumerate(merge):
    if e % 500 == 0:
      print(e, m)
    if db.get(bytes(m, 'utf8')) is not None:

      continue
    with open("../PixivImageScraper/metas/{}".format(m)) as mf, open("./vectors/{}".format(m)) as vf:
      metas = json.loads(mf.read())
      vec   = json.loads(vf.read())
      metas["v"] = vec
    db.put(bytes(m, 'utf8'), pickle.dumps(metas))

""" vectorを取り出してkmeans """
def step3():
  db = plyvel.DB('userid_metavec.ldb', create_if_missing=False)
  vs = []
  for e, (userid, metavec) in enumerate(db):
    if e % 500 == 0:
      print("now iter ", e)
    if e > 50000:
      break
    metavec = pickle.loads(metavec)
    vs.append( metavec["v"] )
  vs = np.array( vs )
  print("finished make numpy")
  kmeans = KMeans(n_clusters=1024, random_state=0)
  print("fit by kmeans")
  kmeans.fit(vs)
  print("fnished fitting by kmeans")
  open("kmeans-1.pkl", "wb").write( pickle.dumps(kmeans) )

""" clustringして、各メタデータをマップ """
def step3():
  db      = plyvel.DB('userid_metavec.ldb', create_if_missing=False)
  keams_1 = pickle.loads( open("kmeans-1.pkl", "rb").read() )
  for e, (userid, metavec) in enumerate(db):
    if e % 500 == 0:
      print("now iter ", e)
    v = pickle.loads(metavec)
    c = keams_1.predict( np.array( [v] ) )
    print(c)

     
if __name__ == '__main__':
  if '--step1' in sys.argv:
    step1()
  if '--step2' in sys.argv:
    step2()
  if '--step3' in sys.argv:
    step3()
