#include "test_class.h"
#include "Bitmap.h"
#include "texture2d.h"
#include "sampler.h"
#include <cassert>

void SimpleCompressColor(float4* color)
{
  color->x /= (1.0F + color->x);
  color->y /= (1.0F + color->y);
  color->z /= (1.0F + color->z);
}

uint RealColorToUint32(float4 a_realColor, const float a_gamma)
{
  float  r = pow(clamp(a_realColor.x, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  g = pow(clamp(a_realColor.y, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  b = pow(clamp(a_realColor.z, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  a =     clamp(a_realColor.w, 0.0F, 1.0F)           * 255.0f;

  unsigned char red   = (unsigned char)r;
  unsigned char green = (unsigned char)g;
  unsigned char blue  = (unsigned char)b;
  unsigned char alpha = (unsigned char)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ToneMapping::ToneMapping(const int w, const int h): m_width(w), m_height(h), m_widthSmall(w/4), m_heightSmall(h/4)
{
  // init weights for gaussian blur
  //
  m_blurRadius    = 100;
  m_filterWeights = createGaussKernelWeights1D_HDRImage(m_blurRadius*2 + 1, 1.25f);
        
  m_brightPixels.resize(w, h);
  m_downsampledImage.resize(m_widthSmall, m_heightSmall);
  m_tempImage.resize(m_widthSmall, m_heightSmall);

  m_sampler.filter = Sampler::Filter::MIN_MAG_LINEAR_MIP_POINT; 
}


void ToneMapping::Bloom(const int a_width, const int a_height, const Texture2D<float4>& a_texture2d, unsigned int* outData1ui)
{
  // (1) ExtractBrightPixels (inData4f => m_brightPixels (w,h))
  //
  kernel2D_ExtractBrightPixels(a_width, a_height, a_texture2d, m_brightPixels);

  // (2) Downsample (m_brightPixels => m_downsampledImage (w/4, h/4) )
  //
  kernel2D_DownSample4x(m_widthSmall, m_heightSmall, m_brightPixels, m_downsampledImage);

  // (3) GaussBlur (m_downsampledImage => m_downsampledImage)
  //
  kernel2D_BlurX(m_widthSmall, m_heightSmall, m_downsampledImage, m_tempImage); // m_downsampledImage => m_tempImage
  kernel2D_BlurY(m_widthSmall, m_heightSmall, m_tempImage, m_downsampledImage); // m_tempImage => m_downsampledImage

  // (4) MixAndToneMap(inData4f, m_downsampledImage) => outData1ui
  //
  kernel2D_MixAndToneMap(a_width, a_height, a_texture2d, m_downsampledImage, outData1ui);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ToneMapping::kernel2D_ExtractBrightPixels(const int a_width, const int a_height, const Texture2D<float4>& a_texture2d, Texture2D<float4>& a_brightPixels)
{  
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {       
      const uint  linearCoord = pitch(x, y, a_width);
      const uint2 coord(x, y);

      float4 color = a_texture2d[coord];
      if(color.x < 1.0f || color.y < 1.0f || color.z < 1.0f)
        color = float4(0.0f, 0.0f, 0.0f, 0.0f);      
        
      a_brightPixels[coord] = color;
    }
  }
}


void ToneMapping::kernel2D_DownSample4x(const int a_width, const int a_height, 
                                        const Texture2D<float4>& a_texture2dFullRes, Texture2D<float4>& a_dataSmallRes)
{
  #pragma omp parallel for
  for(int j = 0; j < a_height; j++)
  {
    for(int i = 0; i < a_width; i++)
    {
      float4 average = float4(0, 0, 0, 0);      

      for(int y = 0; y < 4; y++)
      {
        for(int x = 0; x < 4; x++)
          average += a_texture2dFullRes[uint2(i*4 + x, j*4 + y)];
      }

      const uint2 coord(i, j);
      a_dataSmallRes[coord] = average * (1.0f/16.0f);      
    }
  }
}


void ToneMapping::kernel2D_BlurX(const int a_width, const int a_height, 
                                 const Texture2D<float4>& a_texture2d, Texture2D<float4>& a_dataOut)
{
  #pragma omp parallel for
  for(int tidY = 0; tidY < a_height; tidY++)
  {
    for(int tidX = 0; tidX < a_width; tidX++)
    {
      const float2 uv1  = get_uv(tidX, tidY, a_width, a_height);      
      float4       summ = m_filterWeights[m_blurRadius] * a_texture2d.sample(m_sampler, uv1);
     
      for (int wid = 1; wid < m_blurRadius; wid++) //  <--- * --->
      {
        int left  = tidX - wid;
        int right = tidX + wid;
    
        if (left  < 0)      left  = 0;
        if (right >= a_width) right = a_width - 1;
    
        const float2 uv2 = get_uv(left,  tidY, a_width, a_height);
        const float2 uv3 = get_uv(right, tidY, a_width, a_height);
        float4 p0 = m_filterWeights[wid + m_blurRadius] * a_texture2d.sample(m_sampler, uv2);
        float4 p1 = m_filterWeights[wid + m_blurRadius] * a_texture2d.sample(m_sampler, uv3);

        summ += (p0 + p1);
      }
    
      a_dataOut[uint2(tidX, tidY)] = summ;
    }
  }
}



void ToneMapping::kernel2D_BlurY(const int a_width, const int a_height, 
                                 const Texture2D<float4>& a_texture2d, Texture2D<float4>& a_dataOut)
{
  #pragma omp parallel for
  for(int tidY = 0; tidY < a_height; tidY++)
  {
    for(int tidX = 0; tidX < a_width; tidX++)
    {
      const float2 uv1 = get_uv(tidX, tidY, a_width, a_height);
      float4 summ      = m_filterWeights[m_blurRadius]*a_texture2d.sample(m_sampler, uv1);
     
      for (int wid = 1; wid < m_blurRadius; wid++) //  <--- * --->
      {
        int left  = tidY-wid;
        int right = tidY+wid;
    
        if(left < 0) left = 0;
        if(right >= m_heightSmall) right = m_heightSmall-1;
    
        const float2 uv2 = get_uv(tidX, left,  a_width, a_height);
        const float2 uv3 = get_uv(tidX, right, a_width, a_height);
        float4 p0 = m_filterWeights[wid + m_blurRadius] * a_texture2d.sample(m_sampler, uv2);
        float4 p1 = m_filterWeights[wid + m_blurRadius] * a_texture2d.sample(m_sampler, uv3);
        summ += (p0 + p1);
      }
    
      a_dataOut[uint2(tidX, tidY)] = summ;
    }
  }
}



void ToneMapping::kernel2D_MixAndToneMap(const int a_width, const int a_height, 
                                         const Texture2D<float4>& a_texture2d, const Texture2D<float4>& inBrightPixels,
                                         unsigned int* outData1ui)
{
  #pragma omp parallel for
  for(int tidY = 0; tidY < a_height; tidY++)
  {
    for(int tidX = 0; tidX < a_width; tidX++)
    {
      const float2 uv         = get_uv(tidX, tidY, a_width, a_height);
      const float4 bloomColor = inBrightPixels.sample(m_sampler, uv);
      float4       colorSumm  = bloomColor + a_texture2d.sample(m_sampler, uv);
      
      SimpleCompressColor(&colorSumm);
    
      outData1ui[pitch(tidX, tidY, a_width)] = RealColorToUint32(colorSumm, 1.0F / m_gamma);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SaveTestImage(const float4* data, int w, int h)
{
  std::vector<uint> ldrData(w * h);

  #pragma omp parallel for
  for(size_t i = 0; i < w * h; i++)
    ldrData[i] = RealColorToUint32(clamp(data[i], 0.0f, 1.0f), 1.0F / 2.2F);

  SaveBMP("ztest.bmp", ldrData.data(), w, h);
}

void tone_mapping_cpu(int w, int h, const float* a_hdrData, const char* a_outName)
{
  Texture2D<float4> texture(w, h, (const float4*)a_hdrData);
  std::vector<uint> ldrData(w*h);

  ToneMapping filter(w, h);  
  filter.Bloom(w, h, texture, ldrData.data());
  
  SaveBMP(a_outName, ldrData.data(), w, h);  
}

