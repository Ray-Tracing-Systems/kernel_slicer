#!/usr/bin/env python3
"""
Generates scenes 'objects_N.xml' (N = 4, 16, 64) for the hybrid rendering sample: a grid of k x k bunnies and teapots
on the ground in front of the camera. Objects are scaled by 1/k, so they cover a similar part of the screen and
triangles become smaller when there are more objects. Meshes, materials and the ground plane are taken from
'resources/msu-graphics-group/scenes/01_simple_scenes/instanced_objects.xml'.

Run from this folder: python3 make_scenes.py
"""
import math
import os
import random
import re

SRC_DIR  = "../../resources/msu-graphics-group/scenes/01_simple_scenes"
SRC      = os.path.join(SRC_DIR, "instanced_objects.xml")
GROUND_Y = -4.0
CAM_POS  = (0.0, 12.0, 22.0)
CAM_LOOK = (0.0, -4.0, -2.0)

# mesh id: (bounding box min y, size along x, remap lists from the source scene)
MESHES = {
  0: (0.021,  1.60, [6, 7, 9, 10, 11]),  # bunny: white, green, white, gold, red (glass is skipped)
  1: (-0.405, 2.04, [1, 2, 3, 4, 5]),    # teapot: gold, white, green, white, red
}

def instance(inst_id, mesh_id, rmap_id, matrix):
  vals = " ".join("%g" % v for v in matrix)
  return '    <instance id="%d" mesh_id="%d" rmap_id="%d" scn_id="0" scn_sid="0" matrix="%s " />' % (inst_id, mesh_id, rmap_id, vals)

def make(src_text, k):
  rnd   = random.Random(k)
  lines = []
  depth = 20.0          # rows from z = 8 to z = -12
  for row in range(k):
    z     = 8.0 - (row + 0.5)*depth/k
    dist  = math.hypot(CAM_POS[2] - z, CAM_POS[1] - GROUND_Y)
    halfW = 0.8*dist*math.tan(math.radians(22.5))*16.0/9.0   # 80% of the visible half width at this distance
    for col in range(k):
      mesh = (row + col) % 2
      minY, sizeX, remaps = MESHES[mesh]
      cell  = 2.0*halfW/k
      scale = min(0.7*cell, 0.8*depth/k)/sizeX
      x     = -halfW + (col + 0.5)*cell
      angle = rnd.uniform(0.0, 2.0*math.pi)
      c, s  = math.cos(angle)*scale, math.sin(angle)*scale
      y     = GROUND_Y - minY*scale
      matrix = [c, 0, s, x,  0, scale, 0, y,  -s, 0, c, z,  0, 0, 0, 1]
      lines.append(instance(len(lines), mesh, remaps[(row*k + col) % len(remaps)], matrix))
  lines.append(instance(len(lines), 2, -1, [1, 0, 0, 0,  0, 1, 0, GROUND_Y,  0, 0, 1, 0,  0, 0, 0, 1]))

  text = re.sub(r'loc="data/', 'loc="%s/data/' % SRC_DIR, src_text)  # relative to this folder
  text = re.sub(r'([ \t]*<instance id=[^\n]*\n)+', "\n".join(lines) + "\n", text, count=1)
  text = re.sub(r'<instance_light[^\n]*\n', "", text)
  text = re.sub(r'<position>.*</position>', '<position>%g %g %g</position>' % CAM_POS, text, count=1)
  text = re.sub(r'<look_at>.*</look_at>', '<look_at>%g %g %g</look_at>' % CAM_LOOK, text, count=1)
  return text

if __name__ == "__main__":
  src = open(SRC).read()
  for k in (2, 4, 8):
    name = "objects_%d.xml" % (k*k)
    open(name, "w").write(make(src, k))
    print("written", name)
