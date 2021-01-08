import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import struct

FILENAME_PREFIX = "BodiesDumpCPU_"

filenames = [filename for filename in os.listdir("dumps") if filename.startswith(FILENAME_PREFIX)]
filenames.sort(key=lambda x: int(x[len(FILENAME_PREFIX):]))

frames = []
RESOLUTION = 512
colors = []
for filename in filenames:
  with open("dumps/" + filename, "rb") as fin:
    bodies_count = struct.unpack("I", fin.read(4))[0]
    if len(colors) == 0:
      colors = np.random.rand(bodies_count, 3)
    frame = np.zeros((RESOLUTION, RESOLUTION, 3), dtype="uint8")
    for i in range(bodies_count):
      body = struct.unpack("ffff", fin.read(16))
      x = body[0]
      y = body[1]
      x = int((x + 2) / 4 * RESOLUTION)
      y = int((y + 2) / 4 * RESOLUTION)
      if x >= 512 or y >= 512 or x < 0 or y < 0:
        continue
      RADIUS = 3
      for h in range(-RADIUS, RADIUS + 1):
        for j in range(-RADIUS, RADIUS + 1):
          x1 = min(max(x + h, 0), 511)
          y1 = min(max(y + j, 0), 511)
          color = (255 * max(1 - (h * h + j * j) / RADIUS, 0) * colors[i])
          frame[x1, y1] += color.astype("uint8")
    frames.append(frame)

imageio.mimsave('bodies_CPU.gif', frames)
