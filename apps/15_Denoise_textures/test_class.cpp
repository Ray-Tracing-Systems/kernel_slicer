#include "test_class.h"
#include "../14_filter_bloom_textures/Bitmap.h"
#include <cassert>

inline uint pitch(uint x, uint y, uint pitch) { return y * pitch + x; } 

inline float2 get_uv(const int x, const int y, const uint width, const uint height)
{
  const float u = (float)(x) / (float)(width);
  const float v = (float)(y) / (float)(height);
  return make_float2(u, v);
}

inline uint RealColorToUint32(float4 a_realColor, const float a_gamma)
{
  float  r = pow(clamp(a_realColor.x, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  g = pow(clamp(a_realColor.y, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  b = pow(clamp(a_realColor.z, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  a =     clamp(a_realColor.w, 0.0F, 1.0F)           * 255.0f;

  uint red   = (uint)r;
  uint green = (uint)g;
  uint blue  = (uint)b;
  uint alpha = (uint)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}


static inline float Sqrf(const float x) { return x*x; }

static inline int Clampi(const int x, const int a, const int b)
{  
  if      (x < a) return a;
  else if (x > b) return b;
  else            return x;
}

struct PixelLdr { unsigned char r, g, b; };

PixelLdr DecodeIntToInt3(const int32_t& pxData)
{
  PixelLdr pix;
  pix.r = (pxData & 0x000000FF);
  pix.g = (pxData & 0x0000FF00) >> 8;
  pix.b = (pxData & 0x00FF0000) >> 16;
    
  return pix;
}

static void SimpleCompressColor(float4* color)
{
  color->x /= (1.0F + color->x);
  color->y /= (1.0F + color->y);
  color->z /= (1.0F + color->z);
}


static inline float ProjectedPixelSize(const float dist, const float FOV, const float w, const float h)
{
  float ppx = (FOV / w)*dist;
  float ppy = (FOV / h)*dist;

  if (dist > 0.0F) return 2.0F * fmax(ppx, ppy);
  else             return 1000.0F;
}


static inline float SurfaceSimilarity(const float4 data1, const float4 data2,const float MADXDIFF)
{
  const float MANXDIFF = 0.1F;

  const float3 n1      = to_float3(data1);
  const float3 n2      = to_float3(data2);

  const float dist = length(n1 - n2);
  if (dist >= MANXDIFF)
    return 0.0F;

  const float d1 = data1.w;
  const float d2 = data2.w;

  if (fabs(d1 - d2) >= MADXDIFF)
    return 0.0F;

  const float normalDiff = sqrt(1.0F -         (dist / MANXDIFF));
  const float depthDiff  = sqrt(1.0F - fabs(d1 - d2) / MADXDIFF);

  return normalDiff * depthDiff;
}

static void SaveTestImage(const float4* data, int w, int h)
{
  size_t sizeImg = w * h;
  std::vector<uint> ldrData(sizeImg);

  #pragma omp parallel for
  for(size_t i = 0; i < sizeImg; ++i)
    ldrData[i] = RealColorToUint32(data[i], 1.0F / 2.2F);

  SaveBMP("ztest.bmp", ldrData.data(), w, h);
}

void Denoise::PrepareInput(int w, int h, const float4* in_color, const int32_t* a_inTexColor, const int32_t* a_inNormal, const float4* a_inDepth)
{
  m_width   = w;
  m_height  = h;
  m_sizeImg = w * h;  

  m_texColor.resize (w, h);
  m_normDepth.resize(w, h);
  m_hdrColor.resize (w, h);

  #pragma omp parallel for
  for (size_t y = 0; y < m_height; ++y)  
  { 
    for (size_t x = 0; x < m_width; ++x)
    {      
      const int2 coord(x, y);
      const uint linearCoord = pitch(x, y, m_width);

      // beauty pass
      m_hdrColor[coord] = in_color[linearCoord];

      // color/albedo pass
      PixelLdr ldrColor = DecodeIntToInt3(a_inTexColor[linearCoord]);

      float4 color;    
      color.x = powf((float)ldrColor.r / 255.0F, m_gamma);
      color.y = powf((float)ldrColor.g / 255.0F, m_gamma);
      color.z = powf((float)ldrColor.b / 255.0F, m_gamma);
      color.w = 0.0F;

      m_texColor[coord] = color;

      // Normal and depth pass    
      ldrColor = DecodeIntToInt3(a_inNormal[linearCoord]);

      color.x = powf((float)ldrColor.r / 255.0F, m_gamma) * 2.0F - 1.0F;
      color.y = powf((float)ldrColor.g / 255.0F, m_gamma) * 2.0F - 1.0F;
      color.z = powf((float)ldrColor.b / 255.0F, m_gamma) * 2.0F - 1.0F;
      color.w = a_inDepth[linearCoord].x;      
    
      m_normDepth[coord] = color;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float Denoise::NLMWeight(const Texture2D<float4>& a_texture, int w, int h, int x, int y, int x1, int y1, int a_blockRadius)
{
  float w1        = 0.0f;  // this is what NLM differs from KNN (bilateral)
  const int minX1 = Clampi(x1 - a_blockRadius, 0, w - 1);
  const int maxX1 = Clampi(x1 + a_blockRadius, 0, w - 1);
  const int minY1 = Clampi(y1 - a_blockRadius, 0, h - 1);
  const int maxY1 = Clampi(y1 + a_blockRadius, 0, h - 1);

  for (int y2 = minY1; y2 <= maxY1; ++y2)
  {
    for (int x2 = minX1; x2 <= maxX1; ++x2)
    {
      const int offsX   = x2 - x1;
      const int offsY   = y2 - y1;
      const int x3      = x + offsX;
      const int y3      = y + offsY;
  
      const float2 uv2  = get_uv(x2, y2, w, h);
      const float2 uv3  = get_uv(x3, y3, w, h);
      const float4 c2   = a_texture.sample(m_sampler, uv2);
      const float4 c3   = a_texture.sample(m_sampler, uv3);

      const float4 dist = c2 - c3;
      w1               += dot(dist, dist);
    }
  }

  return w1 / Sqrf(2.0F * (float)a_blockRadius + 1.0F);
}


void Denoise::kernel2D_GuidedTexNormDepthDenoise(const int a_width, const int a_height, unsigned int* a_outData1ui, const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel)
{     
  m_noiseLevel  = 1.0f / (a_noiseLevel * a_noiseLevel);    
  m_windowArea  = Sqrf(2.0f * (float)(a_windowRadius) + 1.0f);

  #pragma omp parallel for
  for (int y = 0; y < a_height; ++y)
  {
    for (int x = 0; x < a_width; ++x)
    {
      const int minX      = Clampi(x - a_windowRadius, 0, a_width - 1);
      const int maxX      = Clampi(x + a_windowRadius, 0, a_width - 1);
      const int minY      = Clampi(y - a_windowRadius, 0, a_height - 1);
      const int maxY      = Clampi(y + a_windowRadius, 0, a_height - 1);

      const float2 uv0    = get_uv(x, y, a_width, a_height);
      const float4 c0     = m_hdrColor.sample(m_sampler, uv0);
      const float4 n0     = m_normDepth.sample(m_sampler, uv0);

      const float ppSize  = 1.0F * (float)(a_windowRadius) * ProjectedPixelSize(n0.w, m_fov, (float)(a_width), (float)(a_height));
      int counterPass     = 0;
      float fSum          = 0.0F;
      float4 result       = float4(0.0F, 0.0F, 0.0F, 0.0F);

      // do window
      //
      for (int y1 = minY; y1 <= maxY; ++y1)
      {
        for (int x1 = minX; x1 <= maxX; ++x1)
        {
          const float2 uv1  = get_uv(x1, y1, a_width, a_height);
          const float4 c1   = m_hdrColor.sample(m_sampler, uv1);
          const float4 n1   = m_normDepth.sample(m_sampler, uv1);
          const int i       = x1 - x;
          const int j       = y1 - y;

          const float match = SurfaceSimilarity(n0, n1, ppSize);
          const float w1    = NLMWeight(m_hdrColor, a_width, a_height, x, y, x1, y1, a_blockRadius);
          const float wt    = NLMWeight(m_texColor, a_width, a_height, x, y, x1, y1, a_blockRadius);

          const float w2 = exp(-(w1*m_noiseLevel + (i * i + j * j) * m_gaussianSigma));
          const float w3 = exp(-(wt*m_noiseLevel + (i * i + j * j) * m_gaussianSigma));

          const float wx = w2*w3*clamp(match, 0.25f, 1.0f);

          if (wx > m_weightThreshold)
            counterPass++;

          fSum += wx;
          result += c1 * wx;
        }
      }

      result = result * (1.0f / fSum);

      //  Now the restored pixel is ready
      //  But maybe the area is actually edgy and so it's better to take the pixel from the original image?	
      //  This test shows if the area is smooth or not
      //
      const float lerpQ = ((float)(counterPass) > (m_counterThreshold * m_windowArea)) ? 1.0f - m_lerpCoefficeint : m_lerpCoefficeint;

      //  This is the last lerp
      //  Most common values for g_LerpCoefficient = [0.85, 1];
      //  So if the area is smooth the result will be
      //  RestoredPixel*0.85 + NoisyImage*0.15
      //  If the area is noisy
      //  RestoredPixel*0.15 + NoisyImage*0.85
      //  That allows to preserve edges more thoroughly
      //
      result = lerp(result, c0, lerpQ);

      SimpleCompressColor(&result);
      a_outData1ui[y * a_width + x] = RealColorToUint32(result, 1.0F / m_gamma);
    }

    #pragma omp critical       
    {
      m_linesDone++;
      if(m_linesDone %2 == 0)
      {
        std::cout << "NLM Denoiser: " << (int)(100.0F * (float)(m_linesDone) / (float)(a_height)) << "% \r";
        std::cout.flush();
      }
    }        
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Denoise::NLM_denoise(const int a_width, const int a_height, unsigned int* a_outData1ui, const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel)
{  
  kernel2D_GuidedTexNormDepthDenoise(a_width, a_height, a_outData1ui, a_windowRadius, a_blockRadius, a_noiseLevel); 
}
