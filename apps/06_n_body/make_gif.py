import imageio
import numpy as np
import struct

FILENAME = "BodiesDumpCPU.bin"
RESOLUTION = 512

frames = []
with open(FILENAME, "rb") as fin:
  bodies_count = struct.unpack("I", fin.read(4))[0]
  iters_count = struct.unpack("I", fin.read(4))[0]
  print("bodies_count = ", bodies_count, ", iters_count = ", iters_count)
  colors = np.random.rand(bodies_count, 3)
  for frameId in range(iters_count):
    frame = np.zeros((RESOLUTION, RESOLUTION, 3), dtype="float")
    for i in range(bodies_count):
      body = struct.unpack("ffff", fin.read(16))
      x = body[0]
      y = body[1]
      weight = body[3]
      x = int((x + 2) / 4 * RESOLUTION)
      y = int((y + 2) / 4 * RESOLUTION)
      if x >= RESOLUTION or y >= RESOLUTION or x < 0 or y < 0:
        continue
      RADIUS = int(abs(weight) + 1)
      c = np.array([1, 0, 0]) if weight > 0 else np.array([0, 1, 0])
      ball_x = np.arange(2 * RADIUS + 1) - RADIUS
      ball_y = ball_x[:, None]
      ball = np.maximum(1 - (ball_y ** 2 + ball_x ** 2) ** 0.5 / RADIUS, 0)[:, :, None] * c
      x_target_l = max(x - RADIUS, 0)
      x_target_r = min(x + RADIUS + 1, RESOLUTION)
      y_target_l = max(y - RADIUS, 0)
      y_target_r = min(y + RADIUS + 1, RESOLUTION)
      x_l_offset = x_target_l - (x - RADIUS)
      x_r_offset = 2 * RADIUS + 1 - (x + RADIUS + 1 - x_target_r)
      y_l_offset = y_target_l - (y - RADIUS)
      y_r_offset = 2 * RADIUS + 1 - (y + RADIUS + 1 - y_target_r)
      frame[x_target_l: x_target_r, y_target_l: y_target_r] += ball[x_l_offset: x_r_offset, y_l_offset: y_r_offset]
    frames.append((np.minimum(frame, 1.0) * 255).astype("uint8"))
    print("progress = ", 100.0 * frameId / iters_count,"%", end='\r', flush=True)

imageio.mimsave('bodies_CPU.gif', frames)
