"""
Script to read pkl file

@author Daniele Filippini
"""

import pickle
import numpy as np
from pprint import pprint

def load_output_from_file(filepath):
  with open('./results.pkl', 'rb') as f:  # rb -> read in binary mode
    output = pickle.load(f)
  return output

res = load_output_from_file("results.pkl")

l = len(res[0][0])
print("LENGHT:", l)
pprint(res[0][0])
print("length of proposals:", len(res))

print("LOOP")
c = 0
for i, item in enumerate(res[0][0]):
  if len(item) > 0:
    print("item", i, ": ", item, " item length: ", len(item))
    c = c+1
    bboxes = item[:, :4]
    scores = np.clip(item[:, 4], 0, 1.0)
    labels = np.zeros_like(scores)
    print("Element:", bboxes, " --- ", scores, " --- ", labels)
print("Num = ", c)