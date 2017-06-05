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
    if e > 25000:
      break
    metavec = pickle.loads(metavec)
    vs.append( metavec["v"] )
  vs = np.array( vs )
  print("finished make numpy")
  kmeans = KMeans(n_clusters=256, random_state=0)
  print("fit by kmeans")
  kmeans.fit(vs)
  print("fnished fitting by kmeans")
  open("kmeans-1.pkl", "wb").write( pickle.dumps(kmeans) )

""" clustringして、各メタデータをマップ """
def step4():
  db      = plyvel.DB('userid_metavec.ldb', create_if_missing=False)
  keams_1 = pickle.loads( open("kmeans-1.pkl", "rb").read() )
 
  def _step4_dataset():
    c_ms    = {}
    for e, (userid, metavec) in enumerate(db):
      if e % 500 == 0:
        print("now iter ", e)
      if e > 250000:
        break
      m = pickle.loads(metavec)
      v = m['v']
      c = keams_1.predict( np.array( [v] ) ).tolist().pop()
      ... #print(c)
      if c_ms.get(c) is None:
        c_ms[c] = []
      c_ms[c].append( m )
   
    open("c_ms.pkl", "wb").write( pickle.dumps(c_ms) )
    return c_ms

  c_ms = _step4_dataset()
  #c_ms = pickle.loads( open("c_ms.pkl", "rb").read() )
  
  for c, ms in sorted(c_ms.items(), key=lambda x:x[0]*-1): 
    print("try cluster {} to convert mini-cluster ".format(c) )
    vs = np.array( list(map(lambda x:x['v'], ms)) )
    try:
      kmeans = KMeans(n_clusters=128, random_state=0)
      kmeans.fit(vs)
    except ValueError as e: 
      try:
        kmeans = KMeans(n_clusters=64, random_state=0)
        kmeans.fit(vs)
      except ValueError as e: 
        try:
          kmeans = KMeans(n_clusters=32, random_state=0)
          kmeans.fit(vs)
        except ValueError as e: 
          try:
            kmeans = KMeans(n_clusters=16, random_state=0)
            kmeans.fit(vs)
          except ValueError as e: 
            try:
              kmeans = KMeans(n_clusters=8, random_state=0)
              kmeans.fit(vs)
            except ValueError as e: 
              try:
                kmeans = KMeans(n_clusters=4, random_state=0)
                kmeans.fit(vs)
              except ValueError as e: 
                try:
                  kmeans = KMeans(n_clusters=2, random_state=0)
                  kmeans.fit(vs)
                except ValueError as e: 
                  try:
                    kmeans = KMeans(n_clusters=1, random_state=0)
                    kmeans.fit(vs)
                  except ValueError as e: 
                    print("Error !!", e)
                    ...
    open("kmeans-2_{}.pkl".format(c), "wb").write( pickle.dumps(kmeans) )
    print("finish {}".format(c) )

""" クラスタをキーとしてKVSに保存 """
def dbsync(chaine, key, val):
  db = plyvel.DB('chaineDbs/{}.ldb'.format(chaine), create_if_missing=True)
  db.put(key, val)
  #db.close()


def step5():
  data_db      = plyvel.DB('userid_metavec.ldb', create_if_missing=True)
  keams_1 = pickle.loads( open("kmeans/kmeans-1.pkl", "rb").read() )

  subclus_kmean = {}

  clus_db       = {}
  for sub in sorted(glob.glob("kmeans/kmeans-2_*.pkl")):
    sub_clus = re.search(r"2_(\d{1,}).pkl", sub).group(1)
    print(sub_clus)
    subclus_kmean[int(sub_clus)] = pickle.loads(open(sub, "rb").read())
    #clus_db[int(sub_clus)]       = plyvel.DB('chaineDbs/{}.ldb'.format(sub_clus), create_if_missing=True)
 
  chaine_key_val = {}
  for e, (userid, metavec) in enumerate(data_db):
    if e % 500 == 0:
      print("now iter ", e)
    if e > 1000000:
      break
    m = pickle.loads(metavec)
    v = m['v']
    c  = keams_1.predict( np.array( [v] ) ).tolist().pop()
    submean = subclus_kmean[c]
    sc = submean.predict( np.array( [v] ) ).tolist().pop()
    chaine = "_".join( map(str, [c,sc]) )
    key    = bytes( ",".join( map(str, v ) ) , 'utf8' )
    val    = pickle.dumps(m)
    if chaine_key_val.get(chaine) is None:
      chaine_key_val[chaine] = {}
    chaine_key_val[chaine][key] = val
  
  c = 0
  for chaine, key_val in chaine_key_val.items():
    db = plyvel.DB('chaineDbs/{}.ldb'.format(chaine), create_if_missing=True)
    for key, val in key_val.items():
      c += 1
      if c % 500 == 0:
        print("now mapping {}".format(c) )
      db.put(key, val)
    
    


     
if __name__ == '__main__':
  if '--step1' in sys.argv:
    step1()
  if '--step2' in sys.argv:
    step2()
  if '--step3' in sys.argv:
    step3()
  if '--step4' in sys.argv:
    step4()
  if '--step5' in sys.argv:
    step5()
