#version 460
#extension GL_NV_cooperative_matrix : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, set = 0) buffer data0 { float A[]; };
layout(binding = 1, set = 0) buffer data1 { float B[]; };
layout(binding = 2, set = 0) buffer data2 { float C[]; };

layout( push_constant ) uniform kernelIntArgs
{
  uint m_A_row_len;
  uint m_sizeX;
  uint m_sizeY;
  uint m_A_offset;
  uint m_B_offset;
  uint m_C_offset;
} kgenArgs;

shared float16_t buf[8 * 8];
shared float bufC[8 * 8];

void main()
{
  const uint lM = 8; const uint lN = 8; const uint lK = 8;
  uvec2 matrixID = uvec2(gl_WorkGroupID);
  uint cRow = lM * matrixID.y;
  uint cCol = lN * matrixID.x;
  fcoopmatNV<16, gl_ScopeWorkgroup, lM, lK> matA;
  fcoopmatNV<16, gl_ScopeWorkgroup, lK, lN> matB;
  fcoopmatNV<32, gl_ScopeWorkgroup, lM, lN> matC = fcoopmatNV<32, gl_ScopeWorkgroup, lM, lN>(0.0f);

  for (uint k = 0; k < kgenArgs.m_A_row_len; k += lK) {
    uint aRow = lM * matrixID.y;
    uint aCol = k;
    buf[gl_LocalInvocationID[1] * 8 + gl_LocalInvocationID[0]] =
      float16_t(A[kgenArgs.m_A_offset + kgenArgs.m_A_row_len * (aRow + gl_LocalInvocationID[1]) + aCol + gl_LocalInvocationID[0]]);
    barrier();
    coopMatLoadNV(matA, buf, 0, 8, false);
    barrier();
    uint bRow = k;
    uint bCol = lN * matrixID.x;
    buf[gl_LocalInvocationID[1] * 8 + gl_LocalInvocationID[0]] =
      float16_t(B[kgenArgs.m_B_offset + kgenArgs.m_A_row_len * (bCol + gl_LocalInvocationID[1]) + bRow + gl_LocalInvocationID[0]]);
    barrier();
    coopMatLoadNV(matB, buf, 0, 8, true);
    barrier();
    matC = coopMatMulAddNV(matA, matB, matC);
    barrier();
  }
  coopMatStoreNV(matC, bufC, 0, 8, false);
  barrier();
  C[kgenArgs.m_C_offset + (cRow + gl_LocalInvocationID[1]) * kgenArgs.m_sizeX + cCol + gl_LocalInvocationID[0]] = bufC[gl_LocalInvocationID[1] * 8 + gl_LocalInvocationID[0]];
}
