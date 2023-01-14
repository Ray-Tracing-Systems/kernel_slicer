#ifndef BASIC_PROJ_LOGIC_H
#define BASIC_PROJ_LOGIC_H

#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif

enum WINDOW_SIZE{WIN_WIDTH = 512, WIN_HEIGHT = 512};

static uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#endif