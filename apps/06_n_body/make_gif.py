import imageio
import numpy as np
import struct

FILENAME = "BodiesDumpCPU.bin"
RESOLUTION = 512

frames = []
with open(FILENAME, "rb") as fin:
  bodies_count = struct.unpack("I", fin.read(4))[0]
  iters_count = struct.unpack("I", fin.read(4))[0]
  colors = np.random.rand(bodies_count, 3)
  for _ in range(iters_count):
    frame = np.zeros((RESOLUTION, RESOLUTION, 3), dtype="uint8")
    for i in range(bodies_count):
      body = struct.unpack("ffff", fin.read(16))
      x = body[0]
      y = body[1]
      x = int((x + 2) / 4 * RESOLUTION)
      y = int((y + 2) / 4 * RESOLUTION)
      if x >= RESOLUTION or y >= RESOLUTION or x < 0 or y < 0:
        continue
      RADIUS = 3
      for h in range(-RADIUS, RADIUS + 1):
        for j in range(-RADIUS, RADIUS + 1):
          x1 = min(max(x + h, 0), RESOLUTION - 1)
          y1 = min(max(y + j, 0), RESOLUTION - 1)
          color = (255 * max(1 - (h * h + j * j) ** 0.5 / RADIUS, 0) * colors[i])
          frame[x1, y1] += color.astype("uint8")
    frames.append(frame)

imageio.mimsave('bodies_CPU.gif', frames)
