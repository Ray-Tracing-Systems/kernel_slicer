#include "test_class.h"
#include "Bitmap.h"
#include <cassert>

/////////////////////////////////////////////////////////////////////////////////

static inline float Sqrf(const float x) { return x*x; }

static inline int Clampi(const int x, const int a, const int b)
{  
  if      (x < a) return a;
  else if (x > b) return b;
  else            return x;
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


static inline float NLMWeight(const Sampler& a_sampler, __global const Texture2D<float4>& a_texture, int w, int h, int x, int y, int x1, int y1, int a_blockRadius)
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
  
      // const float4 c2   = in_buff[y2 * w + x2];
      // const float4 c3   = in_buff[y3 * w + x3];
      const float2 uv2  = get_uv(x2, y2, w, h);
      const float2 uv3  = get_uv(x3, y3, w, h);
      const float4 c2   = a_texture.sample(a_sampler, uv2);
      const float4 c3   = a_texture.sample(a_sampler, uv3);

      const float4 dist = c2 - c3;

      w1               += dot(dist, dist);
    }
  }

  return w1 / Sqrf(2.0F * (float)a_blockRadius + 1.0F);
}

static inline float NLMWeightUchar4(const Sampler& a_sampler, __global const Texture2D<uchar4>& a_texture, int w, int h, int x, int y, int x1, int y1, int a_blockRadius)
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
  
      // const float4 c2   = in_buff[y2 * w + x2];
      // const float4 c3   = in_buff[y3 * w + x3];
      const float2 uv2  = get_uv(x2, y2, w, h);
      const float2 uv3  = get_uv(x3, y3, w, h);
      const float4 c2   = a_texture.sample(a_sampler, uv2);
      const float4 c3   = a_texture.sample(a_sampler, uv3);

      const float4 dist = c2 - c3;

      w1               += dot(dist, dist);
    }
  }

  return w1 / Sqrf(2.0F * (float)a_blockRadius + 1.0F);
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


Denoise::Denoise(const int w, const int h)
{
  Resize(w,h);
}

void Denoise::Resize(int w, int h)
{
  m_width   = w;
  m_height  = h;
  m_sizeImg = w * h;  

  m_texColor.resize(w, h);
  m_normal.resize(w, h);
  m_depth.resize(w, h);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void Denoise::kernel1D_PrepareData(const int32_t* a_inTexColor, const int32_t* a_inNormal, const float4* a_inDepth, 
                                   const float4* a_inImage, Texture2D<float4>& a_texture)
{
#pragma omp parallel for
  for (size_t i = 0; i < m_sizeImg; ++i)  
  { 
    a_texture.write_pixel(i, a_inImage[i]);

    int pxData      = a_inTexColor[i];
    int r           = (pxData & 0x00FF0000) >> 16;
    int g           = (pxData & 0x0000FF00) >> 8;
    int b           = (pxData & 0x000000FF);

    uchar4 color;    
    color.x = pow(r, m_gamma);
    color.y = pow(g, m_gamma);
    color.z = pow(b, m_gamma);
    color.w = 0.0F;

    m_texColor.write_pixel(i, color);
    
    float3 normal;
    pxData          = a_inNormal[i];
    r               = (pxData & 0x00FF0000) >> 16;
    g               = (pxData & 0x0000FF00) >> 8;
    b               = (pxData & 0x000000FF);
    
    normal.x = pow((float)r / 255.0F, m_gamma);
    normal.y = pow((float)g / 255.0F, m_gamma);
    normal.z = pow((float)b / 255.0F, m_gamma);
      
    m_normal.write_pixel(i, encodeNormal(normal));    
    m_depth.write_pixel(i, clamp(a_inDepth[i].x, 0.0F, 1.0F) * ((ushort)~0));
  }
}



void Denoise::kernel2D_GuidedTexNormDepthDenoise(const int a_width, const int a_height, const Sampler& a_sampler, 
                                                 const Texture2D<float4>& a_texture, unsigned int* a_outData1ui,
                                                  const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel)
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
      
      //const float4 c0     = a_inImage  [y * a_width + x];
      //const float4 n0     = m_normDepth[y * a_width + x];
      const float2 uv0    = get_uv(x, y, a_width, a_height);
      const float4 c0     = a_texture.sample(a_sampler, uv0);      
      const float3 normal = decodeNormal(m_normal.read_pixel(y * a_width + x));
      const float  depth  = m_depth.read_pixel(y * a_width + x) / 65535.0F;      
      const float4 n0     = make_float4(normal.x, normal.y, normal.z, depth);
      //const float4 t0   = in_texc[y*w + x];

      const float ppSize  = 1.0F * (float)(a_windowRadius) * ProjectedPixelSize(n0.w, m_fov, (float)(a_width), (float)(a_height));

      int counterPass     = 0;

      float fSum          = 0.0F;
      float4 result       = make_float4(0.0F, 0.0F, 0.0F, 0.0F);

      // do window
      //
      for (int y1 = minY; y1 <= maxY; ++y1)
      {
        for (int x1 = minX; x1 <= maxX; ++x1)
        {
          //const float4 c1     = a_inImage  [y1 * a_width + x1];
          //const float4 n1     = m_normDepth[y1 * a_width + x1];
          const float2 uv1    = get_uv(x1, y1, a_width, a_height);
          const float4 c1     = a_texture.sample(a_sampler, uv1);
          //const float4 n1     = m_normDepth.sample(a_sampler, uv1);
          const float3 normal = decodeNormal(m_normal.sample(a_sampler, uv1));
          const float  depth  = m_depth.sample(a_sampler, uv1);
          const float4 n1     = make_float4(normal.x, normal.y, normal.z, depth);

          //const float4 t1 = in_texc[y1*w + x1];

          const int i       = x1 - x;
          const int j       = y1 - y;

          const float match = SurfaceSimilarity(n0, n1, ppSize);

          const float w1    = NLMWeight(a_sampler, a_texture, a_width, a_height, x, y, x1, y1, a_blockRadius);
          const float wt    = NLMWeightUchar4(a_sampler, m_texColor, a_width, a_height, x, y, x1, y1, a_blockRadius);
          //const float w1  = dot3(c1-c0, c1-c0);
          //const float wt  = dot3(t1-t0, t1-t0);

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
      result = lerpFloat4(result, c0, lerpQ);

      SimpleCompressColor(&result);
      a_outData1ui[y * a_width + x] = RealColorToUint32(result, 1.0F / m_gamma);
    }

    #pragma omp critical       
    {
      m_linesDone++;
      std::cout << "NLM Denoiser: " << (int)(100.0F * (float)(m_linesDone) / (float)(a_height)) << std::endl;
    }        
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void Denoise::NLM_denoise(const int a_width, const int a_height, const float4* a_inImage, const Sampler& a_sampler, 
                          Texture2D<float4>& a_texture, unsigned int* a_outData1ui, const int32_t* a_inTexColor, 
                          const int32_t* a_inNormal, const float4* a_inDepth,  const int a_windowRadius, 
                          const int a_blockRadius, const float a_noiseLevel)
{  

  kernel1D_PrepareData(a_inTexColor, a_inNormal, a_inDepth,a_inImage, a_texture);

  kernel2D_GuidedTexNormDepthDenoise(a_width, a_height, a_sampler, a_texture, a_outData1ui, a_windowRadius, a_blockRadius, a_noiseLevel); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void Denoise_cpu(const int w, const int h, const float* a_hdrData, int32_t* a_inTexColor, const int32_t* a_inNormal, 
                 const float* a_inDepth, const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel, 
                 const char* a_outName)
{
  Denoise filter(w, h);
  Sampler           sampler;  
  Texture2D<float4> texture(w, h);
  std::vector<uint> ldrData(w*h);
  
  filter.NLM_denoise(w, h, (const float4*)a_hdrData, sampler, texture, ldrData.data(), a_inTexColor, a_inNormal, 
                    (const float4*)a_inDepth, a_windowRadius, a_blockRadius, a_noiseLevel);
  
  SaveBMP(a_outName, ldrData.data(), w, h);
  return;
}
