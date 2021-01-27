
#include <cmath>

typedef vec4 float4;
typedef vec3 float3;
typedef vec2 float2;

typedef uvec4 uint4;
typedef uvec3 uint3;
typedef uvec2 uint2;

typedef ivec4 int4;
typedef ivec3 int3;
typedef ivec2 int2;

inline uint get_global_id(uint a_dim) { return glcomp_GlobalInvocationID[a_dim]; }

inline float fmin(float x, float y) { return min(x,y); }
inline float fmax(float x, float y) { return max(x,y); }

#define __global

struct float4x4
{
  inline float4x4()  { identity(); }
  inline explicit float4x4(const float A[16])
  {
    m_col[0] = float4{ A[0], A[4], A[8],  A[12] };
    m_col[1] = float4{ A[1], A[5], A[9],  A[13] };
    m_col[2] = float4{ A[2], A[6], A[10], A[14] };
    m_col[3] = float4{ A[3], A[7], A[11], A[15] };
  }
  inline void identity()
  {
    m_col[0] = float4{ 1.0f, 0.0f, 0.0f, 0.0f };
    m_col[1] = float4{ 0.0f, 1.0f, 0.0f, 0.0f };
    m_col[2] = float4{ 0.0f, 0.0f, 1.0f, 0.0f };
    m_col[3] = float4{ 0.0f, 0.0f, 0.0f, 1.0f };
  }
  inline float4 get_col(int i) const                { return m_col[i]; }
  inline void   set_col(int i, const float4& a_col) { m_col[i] = a_col; }
  inline float4 get_row(int i) const { return float4{ m_col[0][i], m_col[1][i], m_col[2][i], m_col[3][i] }; }
  inline void   set_row(int i, const float4& a_col)
  {
    m_col[0][i] = a_col[0];
    m_col[1][i] = a_col[1];
    m_col[2][i] = a_col[2];
    m_col[3][i] = a_col[3];
  }
  inline float4& col(int i)       { return m_col[i]; }
  inline float4  col(int i) const { return m_col[i]; }
  inline float& operator()(int row, int col)       { return m_col[col][row]; }
  inline float  operator()(int row, int col) const { return m_col[col][row]; }
  float4 m_col[4];
};

