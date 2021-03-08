import numpy as np
import struct
import imageio
import math
import sys

COEFS_COUNT = 9

with open("out.bin", "rb") as fin:
  coefs = np.zeros((COEFS_COUNT, 3))
  for i in range(COEFS_COUNT):
    coefs[i] = np.array([*struct.unpack("fff", fin.read(12))])

plot = np.zeros((512, 512, 3))
for i in range(512):
  for j in range(512):
    y = -(i - 256) / 256
    x = (j - 256) / 256
    if x * x + y * y > 1:
      continue
    z = (1 - x * x - y * y) ** 0.5
    plot[i, j] += coefs[0]
    plot[i, j] += coefs[1] * y
    plot[i, j] += coefs[2] * z
    plot[i, j] += coefs[3] * x
    plot[i, j] += coefs[4] * x * y
    plot[i, j] += coefs[5] * y * z
    plot[i, j] += coefs[6] * (2 * z ** 2 - x ** 2 - y ** 2)
    plot[i, j] += coefs[7] * x * z
    plot[i, j] += coefs[8] * x ** 2 - y ** 2

imageio.imwrite(sys.argv[1] + '_res.png', plot)
