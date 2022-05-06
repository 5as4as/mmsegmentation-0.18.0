import pickle
import numpy as np
import cv2
import os

with open('/home/gengruixiang/anaconda3/envs/mmsegmentation0.18.0/1.pkl', 'rb') as f:
    info = pickle.load(f)

output_dir = 'ddr'

os.makedirs(output_dir, exist_ok=True)

idx = 1
for i in info:
    i[i==1] = 128
    i[i==2] = 255
    cv2.imwrite(os.path.join(output_dir, str(idx) + '.png'), i)
    idx += 1