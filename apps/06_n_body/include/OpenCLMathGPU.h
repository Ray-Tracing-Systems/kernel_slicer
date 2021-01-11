#ifndef OPENCL_MATH_GPU_H
#define OPENCL_MATH_GPU_H

#ifndef MAXFLOAT
#define MAXFLOAT 1e37f
#endif

#define	RAND_MAX	2147483647

typedef uint uint32_t;
typedef int  int32_t;

typedef struct float4x4T
{
  float4 m_col[4];
} float4x4;

#endif
