#include "conv2d.h"
#include <algorithm>
#include <chrono>
#include <cmath>

using LiteMath::clamp;

static inline int clampCoord(int x, int a_size) { return std::max(0, std::min(x, a_size - 1)); } // "clamp to edge" border

static inline float maxAbsRGB(float4 c) { return std::max(std::abs(c.x), std::max(std::abs(c.y), std::abs(c.z))); }

static inline uint32_t PackRGBA8(float4 c)
{
  const uint32_t r = (uint32_t)(clamp(c.x, 0.0f, 1.0f)*255.0f);
  const uint32_t g = (uint32_t)(clamp(c.y, 0.0f, 1.0f)*255.0f);
  const uint32_t b = (uint32_t)(clamp(c.z, 0.0f, 1.0f)*255.0f);
  return 0xFF000000 | (b << 16) | (g << 8) | r;
}

void Convolution2D::SetMaxImageSize(int w, int h)
{
  m_temp.resize(size_t(w)*size_t(h));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////// kernels

void Convolution2D::kernel2D_Convolve(int w, int h, const float4* a_in, const float* a_weights, int a_radius, float4* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      const int kSize = 2*a_radius + 1;
      float4 summ = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for(int ky = -a_radius; ky <= a_radius; ky++)
      {
        const int sy = clampCoord(y + ky, h);
        for(int kx = -a_radius; kx <= a_radius; kx++)
        {
          const int   sx     = clampCoord(x + kx, w);
          const float weight = a_weights[(ky + a_radius)*kSize + (kx + a_radius)];
          summ += a_in[sy*w + sx]*weight;
        }
      }
      a_out[y*w + x] = summ;
    }
  }
}

void Convolution2D::kernel2D_ConvolveRow(int w, int h, const float4* a_in, const float* a_weights, int a_radius, float4* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      float4 summ = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for(int k = -a_radius; k <= a_radius; k++)
        summ += a_in[y*w + clampCoord(x + k, w)]*a_weights[k + a_radius];
      a_out[y*w + x] = summ;
    }
  }
}

void Convolution2D::kernel2D_ConvolveCol(int w, int h, const float4* a_in, const float* a_weights, int a_radius, float4* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      float4 summ = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for(int k = -a_radius; k <= a_radius; k++)
        summ += a_in[clampCoord(y + k, h)*w + x]*a_weights[k + a_radius];
      a_out[y*w + x] = summ;
    }
  }
}

void Convolution2D::kernel1D_FindMaxAbs(int a_size, const float4* a_in)
{
  m_maxValue = 0.0f;                                     // runs once: reduction init
  for(int i = 0; i < a_size; i++)
    m_maxValue = std::max(m_maxValue, maxAbsRGB(a_in[i])); // parallel max-reduction on GPU
}

void Convolution2D::kernel2D_ToLDR(int w, int h, const float4* a_in, int a_normalize, float a_invGamma, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      // Slang back end: members and kernel arguments must not appear directly inside std:: calls,
      // so copy them to locals first (see "Slang back end" in the kernel-slicer skill).
      const float maxValue = m_maxValue;
      const float invGamma = a_invGamma;
      const float4 c = a_in[y*w + x];
      float4 v = float4(std::abs(c.x), std::abs(c.y), std::abs(c.z), 1.0f);
      if(a_normalize != 0)  // log-scale normalization by the reduced maximum: HDR highlights don't hide weak edges
      {
        const float invLogMax = 1.0f/std::log(1.0f + std::max(maxValue, 1e-6f));
        v.x = std::log(1.0f + v.x)*invLogMax;
        v.y = std::log(1.0f + v.y)*invLogMax;
        v.z = std::log(1.0f + v.z)*invLogMax;
      }
      float4 res;
      res.x = std::pow(clamp(v.x, 0.0f, 1.0f), invGamma);
      res.y = std::pow(clamp(v.y, 0.0f, 1.0f), invGamma);
      res.z = std::pow(clamp(v.z, 0.0f, 1.0f), invGamma);
      res.w = 1.0f;
      a_out[y*w + x] = PackRGBA8(res);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////// control functions

void Convolution2D::Convolve2D(int w, int h, const float4* a_in, const float* a_weights, int a_kernelSize, float4* a_out)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_Convolve(w, h, a_in, a_weights, a_kernelSize/2, a_out);
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}

void Convolution2D::ConvolveSeparable(int w, int h, const float4* a_in, const float* a_weightsX, const float* a_weightsY, int a_kernelSize, float4* a_out)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_ConvolveRow(w, h, a_in,          a_weightsX, a_kernelSize/2, m_temp.data()); // a_in   -> m_temp
  kernel2D_ConvolveCol(w, h, m_temp.data(), a_weightsY, a_kernelSize/2, a_out);         // m_temp -> a_out
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}

void Convolution2D::ToLDR(int w, int h, const float4* a_in, int a_normalize, float a_gamma, uint32_t* a_out)
{
  kernel1D_FindMaxAbs(w*h, a_in);
  kernel2D_ToLDR(w, h, a_in, a_normalize, 1.0f/a_gamma, a_out);
}
