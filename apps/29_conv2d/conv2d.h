#pragma once
#include <vector>
#include <cstdint>
#include "LiteMath.h"

using LiteMath::float4;
using LiteMath::uint;

// 2D image convolution with square kernels of size 3x3, 5x5 or 7x7 (any odd size up to MAX_KERNEL_SIZE).
// Three control functions:
//   Convolve2D        - direct 2D convolution, K*K taps per pixel;
//   ConvolveSeparable - separable convolution (rows pass, then columns pass), 2*K taps per pixel;
//   ToLDR             - converts float4 result into displayable 8-bit RGBA, optionally log-normalizing
//                       by the maximum absolute value found with a parallel max-reduction.
class Convolution2D
{
public:
  static constexpr int MAX_KERNEL_SIZE = 7;

  Convolution2D() {}

  void SetMaxImageSize(int w, int h);  // host only, call before CommitDeviceData()

  virtual void Convolve2D(int w, int h, const float4* a_in      [[size("w*h")]],
                                        const float*  a_weights [[size("a_kernelSize*a_kernelSize")]], int a_kernelSize,
                                        float4*       a_out     [[size("w*h")]]);

  virtual void ConvolveSeparable(int w, int h, const float4* a_in       [[size("w*h")]],
                                               const float*  a_weightsX [[size("a_kernelSize")]],
                                               const float*  a_weightsY [[size("a_kernelSize")]], int a_kernelSize,
                                               float4*       a_out      [[size("w*h")]]);

  virtual void ToLDR(int w, int h, const float4* a_in [[size("w*h")]], int a_normalize, float a_gamma,
                     uint32_t* a_out [[size("w*h")]]);

  virtual void CommitDeviceData() {}                                                         // overridden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; } // overridden in generated class

  float GetMaxValue() const { return m_maxValue; }

protected:
  virtual void kernel2D_Convolve   (int w, int h, const float4* a_in, const float* a_weights, int a_radius, float4* a_out);
  virtual void kernel2D_ConvolveRow(int w, int h, const float4* a_in, const float* a_weights, int a_radius, float4* a_out);
  virtual void kernel2D_ConvolveCol(int w, int h, const float4* a_in, const float* a_weights, int a_radius, float4* a_out);
  virtual void kernel1D_FindMaxAbs (int a_size, const float4* a_in);
  virtual void kernel2D_ToLDR      (int w, int h, const float4* a_in, int a_normalize, float a_invGamma, uint32_t* a_out);

  std::vector<float4> m_temp;       // intermediate image between separable passes
  float m_maxValue = 1.0f;          // result of max-reduction, read back to CPU
  float m_time     = 0.0f;
};
