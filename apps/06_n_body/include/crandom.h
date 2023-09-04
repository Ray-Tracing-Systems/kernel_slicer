#ifndef RTC_RANDOM
#define RTC_RANDOM

#include "LiteMath.h"
#ifndef ISPC
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif
#define varying
#endif

typedef struct RandomGenT
{
  uint2 state;

} RandomGen;

static inline uint32_t NextState(varying RandomGen* gen)
{
  const uint32_t x = (gen->state).x * 17 + (gen->state).y * 13123;
  (gen->state).x = (x << 13u) ^ x;
  (gen->state).y ^= (x << 7);
  return x;
}

static inline RandomGen RandomGenInit(const uint32_t a_seed)
{
  RandomGen gen;

  gen.state.x = (a_seed * (a_seed * a_seed * 15731 + 74323) + 871483);
  gen.state.y = (a_seed * (a_seed * a_seed * 13734 + 37828) + 234234);

  for(uint32_t i = 0; i < (a_seed % 7u); i++)
    NextState(&gen);

  return gen;
}

static inline unsigned int rndInt_Pseudo(varying RandomGen* gen)
{
  return NextState(gen);
}

static inline float4 rndFloat4_Pseudo(varying RandomGen* gen)
{
  unsigned int x = NextState(gen);

  const unsigned int x1 = (x * (x * x * 15731 + 74323) + 871483);
  const unsigned int y1 = (x * (x * x * 13734 + 37828) + 234234);
  const unsigned int z1 = (x * (x * x * 11687 + 26461) + 137589);
  const unsigned int w1 = (x * (x * x * 15707 + 789221) + 1376312589);

  const float scale = (1.0f / 4294967296.0f);

  return make_float4((float)(x1), (float)(y1), (float)(z1), (float)(w1))*scale;
}

static inline float2 rndFloat2_Pseudo(varying RandomGen* gen)
{
  unsigned int x = NextState(gen); 

  const unsigned int x1 = (x * (x * x * 15731 + 74323) + 871483);
  const unsigned int y1 = (x * (x * x * 13734 + 37828) + 234234);

  const float scale     = (1.0f / 4294967296.0f);

  return make_float2((float)(x1), (float)(y1))*scale;
}

static inline float rndFloat1_Pseudo(varying RandomGen* gen)
{
  const unsigned int x   = NextState(gen);
  const unsigned int tmp = (x * (x * x * 15731 + 74323) + 871483);
  const float scale      = (1.0f / 4294967296.0f);
  return ((float)(tmp))*scale;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
\brief map float random to int, including boundaries.
\param a_val - in random float in range [0,1].
\param a     - in min value
\param b     - in max value

\return integer random in ragne [a,b]

for example mapRndFloatToInt(r,3,5) will give these numbers: (3,4,5)
for example mapRndFloatToInt(r,1,6) will give these numbers: (1,2,3,4,5,6)

*/
static inline int mapRndFloatToInt(float a_val, int a, int b)
{
  const float fa = (float)(a+0);
  const float fb = (float)(b+1);
  const float fR = fa + a_val * (fb - fa);

  const int res =  (int)(fR);

  if (res > b)
    return b;
  else
    return res;
}

static inline float4 rndUniform(varying RandomGen* gen, float a, float b)
{
  return make_float4(a, a, a, a) + (b - a)*rndFloat4_Pseudo(gen);
}

#endif
