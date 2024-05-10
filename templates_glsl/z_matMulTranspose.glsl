#version 460
#extension GL_NV_cooperative_matrix : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

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

const uint C_ROWS = 2;
const uint C_COLS = 2;
shared float16_t buf[8 * 8];
shared float bufC[8 * 8];

void main()
{
  const uint lM = 8; const uint lN = 8; const uint lK = 8;
  uvec2 matrixID = uvec2(gl_WorkGroupID);
  fcoopmatNV<32, gl_ScopeWorkgroup, lM, lN> matC[C_ROWS * C_COLS];
  for (uint j = 0; j < C_COLS; ++j) {
    for (uint i = 0; i < C_ROWS; ++i) {
      matC[i * C_COLS + j] = fcoopmatNV<32, gl_ScopeWorkgroup, lM, lN>(0.0f);
      barrier();
    }
  }
  barrier();

  uint localX = gl_LocalInvocationID[0] / 8;
  uint localY = gl_LocalInvocationID[0] % 8;

  for (uint k = 0; k < kgenArgs.m_A_row_len; k += lK) {
    fcoopmatNV<16, gl_ScopeWorkgroup, lM, lK> matA[C_ROWS];
    for (uint i = 0; i < C_ROWS; ++i) {
      uint aRow = lM * (C_ROWS * matrixID.y + i);
      uint aCol = k;
      buf[gl_LocalInvocationID[0]] =
        float16_t(A[kgenArgs.m_A_offset + kgenArgs.m_A_row_len * (aRow + localX) + aCol + localY]);
      barrier();
      coopMatLoadNV(matA[i], buf, 0, 8, false);
      barrier();
    }

    for (uint j = 0; j < C_COLS; ++j) {
      uint bRow = k;
      uint bCol = lN * (matrixID.x * C_COLS + j);

      buf[gl_LocalInvocationID[0]] =
        float16_t(B[kgenArgs.m_B_offset + kgenArgs.m_A_row_len * (bCol + localX) + bRow + localY]);
      barrier();
      fcoopmatNV<16, gl_ScopeWorkgroup, lK, lN> matB;
      coopMatLoadNV(matB, buf, 0, 8, true);
      barrier();

      for (uint i = 0; i < C_ROWS; ++i) {
        matC[(i * C_COLS + j)] = coopMatMulAddNV(matA[i], matB, matC[(i * C_COLS + j)]);
        barrier();
      }
    }
  }
  barrier();

  for (uint j = 0; j < C_COLS; ++j) {
    for (uint i = 0; i < C_ROWS; ++i) {
      coopMatStoreNV(matC[(i * C_COLS + j)], bufC, 0, 8, false);
      barrier();
      C[kgenArgs.m_C_offset + ((lM * (C_ROWS * matrixID.y + i)) + localX) * kgenArgs.m_sizeX + lN * (matrixID.x * C_COLS + j) + localY] = bufC[gl_LocalInvocationID[0]];
      barrier();
    }
  }
}
