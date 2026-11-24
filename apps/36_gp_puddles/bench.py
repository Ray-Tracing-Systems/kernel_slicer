#!/usr/bin/env python3
"""
Measures the hybrid rendering sample on CPU (input C++ code), ISPC and GPU for several scenes and writes bench.json.
Expects binaries build/cpu/testapp, build/ispc/testapp (-DUSE_ISPC=ON) and build/gpu/testapp (Vulkan).
Run from this folder: python3 bench.py [--width 3840 --height 2160]
"""
import argparse
import json
import os
import re
import shutil
import subprocess

SCENES = [("4",   "scenes/objects_4.xml"),
          ("16",  "scenes/objects_16.xml"),
          ("64",  "scenes/objects_64.xml"),
          ("600", "../resources/msu-graphics-group/scenes/01_simple_scenes/instanced_objects.xml")]

# name, executable, options, runs (small scenes, 600 objects), OpenMP threads (None: all), image
VARIANTS = [("cpu",     "build/cpu/testapp",  [],         (10, 3), None, "zout_cpu.bmp"),
            ("ispc",    "build/ispc/testapp", ["--ispc"], (10, 5), None, "zout_ispc.bmp"),
            ("gpu",     "build/gpu/testapp",  ["--gpu"],  (50, 50), None, "zout_gpu.bmp"),
            ("cpu_1t",  "build/cpu/testapp",  [],         (3, 0),  1,    "zout_cpu.bmp"),   # one thread: gain of vectorization
            ("ispc_1t", "build/ispc/testapp", ["--ispc"], (3, 0),  1,    "zout_ispc.bmp")]

def parse(out):
  res = {}
  def num(pattern):
    m = re.search(pattern, out)
    return float(m.group(1)) if m else None
  res["gbuffer_wall"] = num(r"RenderGBuffer\s+= ([0-9.e+-]+) ms")
  res["gbuffer_exec"] = num(r"RenderGBuffer\s+= [0-9.e+-]+ ms \(GetExecutionTime: ([0-9.e+-]+) ms")
  res["shade_wall"]   = num(r"ShadeAndReflect = ([0-9.e+-]+) ms")
  res["shade_exec"]   = num(r"ShadeAndReflect = [0-9.e+-]+ ms \(GetExecutionTime: ([0-9.e+-]+) ms")
  res["vs"]           = num(r"vertex shader = ([0-9.e+-]+) ms")
  res["setup"]        = num(r"setup, binning= ([0-9.e+-]+) ms")
  res["raster"]       = num(r"raster and PS = ([0-9.e+-]+) ms")
  res["triangles"]    = num(r"triangles\s+= ([0-9]+) after setup")
  tris = re.findall(r"mesh \d+ '[^']*': (\d+) triangles, (\d+) instances", out)
  res["input_triangles"] = sum(int(t)*int(n) for t, n in tris)
  return res

def psnr(a, b):
  from PIL import Image, ImageChops, ImageStat
  import math
  d   = ImageChops.difference(Image.open(a).convert("RGB"), Image.open(b).convert("RGB"))
  mse = sum(x*x for x in ImageStat.Stat(d).rms)/3.0
  return float("inf") if mse == 0 else 10.0*math.log10(255.0**2/mse)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--width",  type=int, default=3840)
  ap.add_argument("--height", type=int, default=2160)
  ap.add_argument("--out",    default="bench.json")
  args = ap.parse_args()

  results = {}
  for name, scene in SCENES:
    results[name] = {}
    for var, exe, opts, runsBySize, threads, image in VARIANTS:
      runs = runsBySize[1] if name == "600" else runsBySize[0]
      if runs == 0:
        continue
      env = dict(os.environ)
      if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
      cmd = [exe] + opts + ["--scene", scene, "--width", str(args.width), "--height", str(args.height), "--runs", str(runs), "--stats"]
      out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
      results[name][var] = parse(out)
      shutil.copy(image, "zbench_%s_%s.bmp" % (name, var))
      print(name, var, results[name][var], flush=True)
    results[name]["psnr_ispc"] = psnr("zbench_%s_cpu.bmp" % name, "zbench_%s_ispc.bmp" % name)
    results[name]["psnr_gpu"]  = psnr("zbench_%s_cpu.bmp" % name, "zbench_%s_gpu.bmp" % name)
    print(name, "PSNR vs C++: ispc", results[name]["psnr_ispc"], "gpu", results[name]["psnr_gpu"], flush=True)
  json.dump({"width": args.width, "height": args.height, "results": results}, open(args.out, "w"), indent=2)
