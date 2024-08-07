#!/bin/sh
glslangValidator -V --target-env vulkan1.2 -S rgen BFRT_ReadAndComputeMegaRGEN.glsl -o BFRT_ReadAndComputeMegaRGEN.glsl.spv -DGLSL -I.. -I/home/frol/PROG/kernel_slicer/TINYSTL -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux -I/home/frol/PROG/kernel_slicer/apps/LiteMath 

glslangValidator -V --target-env vulkan1.2 -S rchit z_trace_rchit.glsl -o z_trace_rchit.glsl.spv
glslangValidator -V --target-env vulkan1.2 -S rmiss z_trace_rmiss.glsl -o z_trace_rmiss.glsl.spv

glslangValidator -V --target-env vulkan1.2 -S rchit z_SpherePrim_rchit.glsl -o z_SpherePrim_rchit.glsl.spv
glslangValidator -V --target-env vulkan1.2 -S rint  z_SpherePrim_rcint.glsl -o z_SpherePrim_rcint.glsl.spv
