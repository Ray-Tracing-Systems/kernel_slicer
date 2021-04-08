#include "test_class.h"
#include "Bitmap.h"
#include <cassert>

/////////////////////////////////////////////////////////////////////////////////

inline static uint pitch(uint x, uint y, uint pitch) { return y*pitch + x; }  

static inline float Sqrf(const float x) { return x*x; }

static inline int Clampi(const int x, const int a, const int b)
{  
  if      (x < a) return a;
  else if (x > b) return b;
  else            return x;
}

void SimpleCompressColor(float4* color)
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

  const float normalDiff = sqrtf(1.0F -         (dist / MANXDIFF));
  const float depthDiff  = sqrtf(1.0F - fabs(d1 - d2) / MADXDIFF);

  return normalDiff * depthDiff;
}


static inline float NLMWeight(const float4* in_buff, int w, int h, int x, int y, int x1, int y1, int a_blockRadius)
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
  
      const int x3      = Clampi(x + offsX, 0, w - 1);
      const int y3      = Clampi(y + offsY, 0, h - 1);
  
      const float4 c2   = in_buff[y2 * w + x2];
      const float4 c3   = in_buff[y3 * w + x3];

      const float4 dist = c2 - c3;

      w1               += dot(dist, dist);
    }
  }

  return w1 / Sqrf(2.0F * float(a_blockRadius) + 1.0F);
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


static void Blend(float* inData1, const float inData2, const float amount) // 0 - data1, 1 - data2
{
  *inData1 = *inData1 + (inData2 - *inData1) * amount;
}


void Denoise::SetMaxImageSize(int w, int h)
{
  m_width  = w;
  m_height = h;
  m_size   = w * h;  
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void Denoise::kernel1D_int32toFloat4(const int32_t* a_inTexColor, const int32_t* a_inNormal, const float4* a_inDepth, 
                                     float4* a_texColor, float4* a_normDepth)
{
  #pragma omp parallel for
  for (size_t i = 0; i < m_size; ++i)  
  {      
    int pxData      = a_inTexColor[i];
    int r           = (pxData & 0x00FF0000) >> 16;
    int g           = (pxData & 0x0000FF00) >> 8;
    int b           = (pxData & 0x000000FF);

    a_texColor[i].x = pow((float)r / 255.0F, m_gamma);
    a_texColor[i].y = pow((float)g / 255.0F, m_gamma);
    a_texColor[i].z = pow((float)b / 255.0F, m_gamma);
    a_texColor[i].w = 0.0F;

    pxData          = a_inNormal[i];
    r               = (pxData & 0x00FF0000) >> 16;
    g               = (pxData & 0x0000FF00) >> 8;
    b               = (pxData & 0x000000FF);

    a_normDepth[i].x = pow((float)r / 255.0F, m_gamma);
    a_normDepth[i].y = pow((float)g / 255.0F, m_gamma);
    a_normDepth[i].z = pow((float)b / 255.0F, m_gamma);
    a_normDepth[i].w = a_inDepth[i].x;      
  }
}


void Denoise::kernel2D_GuidedTexNormDepthDenoise(int a_width, const int a_height, const float4* a_inImage, const float4* a_inTexColor, 
const float4* a_inNormDepth, unsigned int* a_outData1ui, const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel)
{     
  const float g_NoiseLevel       = 1.0f / (a_noiseLevel*a_noiseLevel);
  const float g_GaussianSigma    = 1.0f / 50.0f;
  const float g_WeightThreshold  = 0.03f;
  const float g_LerpCoefficeint  = 0.80f;
  const float g_CounterThreshold = 0.05f;

  const float DEG_TO_RAD         = 0.017453292519943295769236907684886f;
  const float m_fov              = DEG_TO_RAD * 90.0f;

  ////////////////////////////////////////////////////////////////////

  const int w            = a_width;
  const int h            = a_height;

  const float4* in_buff  = a_inImage;
  const float4* in_texc  = a_inTexColor;
  const float4* nd_buff  = a_inNormDepth;
  //float4*       out_buff = (float4*)outImage.data();

  const float windowArea = Sqrf(2.0f * float(a_windowRadius) + 1.0f);

  int linesDone          = 0;

  #pragma omp parallel for
  for (int y = 0; y < h; ++y)
  {
    for (int x = 0; x < w; ++x)
    {
      const int minX = Clampi(x - a_windowRadius, 0, w - 1);
      const int maxX = Clampi(x + a_windowRadius, 0, w - 1);

      const int minY = Clampi(y - a_windowRadius, 0, h - 1);
      const int maxY = Clampi(y + a_windowRadius, 0, h - 1);

      const float4 c0 = in_buff[y*w + x];
      const float4 n0 = nd_buff[y*w + x];
      //const float4 t0 = in_texc[y*w + x];

      float ppSize = 1.0F * float(a_windowRadius) * ProjectedPixelSize(n0.w, m_fov, float(w), float(h));

      int counterPass = 0;

      float fSum      = 0.0F;
      float4 result(0, 0, 0, 0);

      // do window
      //
      for (int y1 = minY; y1 <= maxY; ++y1)
      {
        for (int x1 = minX; x1 <= maxX; ++x1)
        {
          const float4 c1   = in_buff[y1*w + x1];
          const float4 n1   = nd_buff[y1*w + x1];
          //const float4 t1 = in_texc[y1*w + x1];

          const int i       = x1 - x;
          const int j       = y1 - y;

          const float match = SurfaceSimilarity(n0, n1, ppSize);

          const float w1    = NLMWeight(in_buff, w, h, x, y, x1, y1, a_blockRadius);
          const float wt    = NLMWeight(in_texc, w, h, x, y, x1, y1, a_blockRadius);
          //const float w1  = dot3(c1-c0, c1-c0);
          //const float wt  = dot3(t1-t0, t1-t0);

          const float w2 = exp(-(w1*g_NoiseLevel + (i * i + j * j) * g_GaussianSigma));
          const float w3 = exp(-(wt*g_NoiseLevel + (i * i + j * j) * g_GaussianSigma));

          const float wx = w2*w3*clamp(match, 0.25f, 1.0f);

          if (wx > g_WeightThreshold)
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
      float lerpQ = (float(counterPass) > (g_CounterThreshold * windowArea)) ? 1.0f - g_LerpCoefficeint : g_LerpCoefficeint;

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
      a_outData1ui[y*w + x] = RealColorToUint32(result, 1.0F / m_gamma);
    }

    #pragma omp critical       
    {
      linesDone++;
      std::cout << "NLM Denoiser: " << int(100.0f*float(linesDone) / float(h)) << std::endl;
    }        
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void Denoise::NLM_denoise(int a_width, const int a_height, const float4* a_inImage, unsigned int* a_outData1ui, const int32_t* a_inTexColor, 
const int32_t* a_inNormal, const float4* a_inDepth,  const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel)
{  
  std::vector<float4> texColor(m_size);
  std::vector<float4> normDepth(m_size);

  kernel1D_int32toFloat4(a_inTexColor, a_inNormal, a_inDepth, texColor.data(), normDepth.data());

  kernel2D_GuidedTexNormDepthDenoise(a_width, a_height, a_inImage, texColor.data(), normDepth.data(),a_outData1ui, 
                                     a_windowRadius, a_blockRadius, a_noiseLevel); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void Denoise_cpu(int w, int h, const float* a_hdrData, int32_t* a_inTexColor, const int32_t* a_inNormal, const float* a_inDepth, 
                 const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel, const char* a_outName)
{
  Denoise filter;
  std::vector<uint> ldrData(w*h);
  
  filter.SetMaxImageSize(w,h);  
  filter.NLM_denoise(w, h, (const float4*)a_hdrData, ldrData.data(), a_inTexColor, a_inNormal, (const float4*)a_inDepth, a_windowRadius,
                     a_blockRadius, a_noiseLevel);
  
  SaveBMP(a_outName, ldrData.data(), w, h);
  return;
}