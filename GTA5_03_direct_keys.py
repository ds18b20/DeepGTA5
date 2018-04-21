# import numpy as np
# from PIL import ImageGrab
# import cv2
import time
from common.directkeys import PressKey, ReleaseKey, W, A, S, D

for i in list(range(5))[::-1]:
    print(i+1)
    time.sleep(1)

print("UP")
PressKey(W)
time.sleep(1)
ReleaseKey(W)
print("DOWN")
PressKey(S)
time.sleep(1)
ReleaseKey(S)
