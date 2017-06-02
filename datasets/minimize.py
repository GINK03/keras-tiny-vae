
import glob
import concurrent.futures

import os
import sys
from PIL import Image
by = 8

def map(name):
  filename = name.split("/").pop()
  im = Image.open(name)
  im = im.resize((28*by,28*by))
  im.save("minimize/{}.png".format(filename))
  print(name)

names = [] 
for name in glob.glob("img_align_celeba/*"):
  names.append(name)
print("finish to load dir ")
with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
  executor.map(map, names)
