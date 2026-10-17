#!/bin/sh
# usage: bash z_build_ispc.sh [ispc target] [embree4 include dir]
#   default target is avx2-i32x8; if embree4 include dir is set, kernels are compiled for Embree 4 (USE_EMBREE4), else for Embree 3 from apps/RayTracing
#   example: bash z_build_ispc.sh avx512skx-x16 $HOME/PROG/embree4/install/include
cd "$(dirname "$0")"
TARGET=${1:-avx2-i32x8}
if [ -n "$2" ]; then
  EMBREE_INC="-I$2 -DUSE_EMBREE4"
else
  EMBREE_INC="-I../../RayTracing"
fi
ISPC=${ISPC:-../../ispc}
$ISPC test_class_kernels.ispc --target=$TARGET -O2 -o test_class_kernels.o -h test_class_kernels.h -I../../LiteMathAux $EMBREE_INC
