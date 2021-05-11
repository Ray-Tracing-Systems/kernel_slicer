#include "test_class.h"
#include "Bitmap.h"
#include "texture2d.h"
#include "sampler.h"
#include <cassert>

inline uint pitch(uint x, uint y, uint pitch) { return y * pitch + x; }  

inline float4 bilinear(__global const float4* data, int w, int h, float a_texCoordX, float a_texCoordY)
{
  const float fw = (float)(w);
  const float fh = (float)(h);

  const float ffx = clamp(a_texCoordX*fw - 0.5f, 0.0f, fw - 1.0f);
  const float ffy = clamp(a_texCoordY*fh - 0.5f, 0.0f, fh - 1.0f);

  const int px = (int)(ffx);
  const int py = (int)(ffy);

  const int stride = w;

  const int px1 = (px < w - 1) ? px + 1 : px;
  const int py1 = (py < h - 1) ? py + 1 : py;

  const int offset0 = (px  + py  * stride);
  const int offset1 = (px1 + py  * stride);
  const int offset2 = (px  + py1 * stride);
  const int offset3 = (px1 + py1 * stride);

  const float  alpha = ffx - (float)px;
  const float  beta  = ffy - (float)py;
  const float  gamma = 1.0f - alpha;
  const float  delta = 1.0f - beta;

  const float4 samplesA = data[offset0]*gamma;
  const float4 samplesB = data[offset1]*alpha;
  const float4 samplesC = data[offset2]*gamma;
  const float4 samplesD = data[offset3]*alpha;

  const float4 resultX0 = samplesA + samplesB;
  const float4 resultX1 = samplesC + samplesD;
  const float4 resultY0 = resultX0*delta;
  const float4 resultY1 = resultX1*beta;

  return resultY0 + resultY1;
}

void SaveTestImage(const float4* data, int w, int h)
{
  std::vector<uint> ldrData(w*h);
  for(size_t i=0;i<w*h;i++)
    ldrData[i] = RealColorToUint32( clamp(data[i], 0.0f, 1.0f));
  SaveBMP("ztest.bmp", ldrData.data(), w, h);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMapping::SetMaxImageSize(int w, int h)
{
  m_width       = w;
  m_height      = h;
  m_widthSmall  = w/4;
  m_heightSmall = h/4;

  m_brightPixels.resize(w*h);
  m_downsampledImage.resize(m_widthSmall*m_heightSmall);
  m_tempImage.resize(m_widthSmall*m_heightSmall);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMapping::kernel2D_ExtractBrightPixels(int width, int height, const Texture2D<float4>* a_texture2d, const Sampler* a_sampler, Texture2D<float4>* a_brightPixels)
{  
  for(int y = 0; y < height; y++)
  {
    for(int x = 0; x < width; x++)
    {
      //float4 pixel = inData4f[pitch(x, y, m_width)];
      const float2 uv        = make_float2(x/(float)(width), y/(float)(height));
      const float4 pixel     = a_texture2d->sample(a_sampler, uv, make_int2(width, height));
      const uint   pos_pixel = pitch(x, y, m_width);

      if(pixel.x >= 1.0f || pixel.y >= 1.0f || pixel.z >= 1.0f)
        a_brightPixels->write_pixel(pos_pixel, pixel);
      else
        a_brightPixels->write_pixel(pos_pixel, make_float4(0,0,0,0));      
    }
  }
}


void ToneMapping::kernel2D_DownSample4x(int width, int height, const Texture2D<float4>* a_dataFullRes, const Sampler* a_sampler, Texture2D<float4>* a_dataSmallRes)
{
  const int2 texSize = make_int2(width, height);

  for(int j = 0; j < height; j++)
  {
    for(int i = 0; i < width; i++)
    {
      float4 average = make_float4(0, 0, 0, 0);      

      for(int y = 0; y < 4; y++)
      {
        for(int x = 0; x < 4; x++)
        {
          const float2 uv = make_float2((float)(i*4 + x) / (float)(width), (float)(j*4 + y) / (float)(height));
          //average += a_dataFullRes->read_pixel(pitch(i*4 + x, j*4 + y, m_width));
          average += a_dataFullRes->sample(a_sampler, uv, texSize);
        }
      }

      a_dataSmallRes->write_pixel(pitch(i, j, m_widthSmall), average*(1.0f/16.0f));      
    }
  }
}


void ToneMapping::kernel2D_BlurX(int width, int height, const Texture2D<float4>* a_dataIn, const Sampler* a_sampler, Texture2D<float4>* a_dataOut)
{
  for(int tidY=0;tidY<height;tidY++)
  {
    for(int tidX=0;tidX<width;tidX++)
    {
      float4 summ = m_filterWeights[m_blurRadius]*a_dataIn[pitch(tidX, tidY, m_widthSmall)]; 
     
      for (int wid = 1; wid < m_blurRadius; wid++) //  <--- * --->
      {
        int left  = tidX-wid;
        int right = tidX+wid;
    
        if(left < 0) left = 0;
        if(right >= m_widthSmall) right = m_widthSmall-1;
    
        float4 p0 = m_filterWeights[wid + m_blurRadius]*a_dataIn[pitch(left, tidY, m_widthSmall)]; 
        float4 p1 = m_filterWeights[wid + m_blurRadius]*a_dataIn[pitch(right, tidY, m_widthSmall)]; 
        summ = summ + (p0 + p1);
      }
    
      a_dataOut[pitch(tidX, tidY, m_widthSmall)] = summ;
    }
  }
}

void ToneMapping::kernel2D_BlurY(int width, int height, const Texture2D<float4>* a_dataIn, const Sampler* a_sampler, Texture2D<float4>* a_dataOut)
{
  for(int tidY=0;tidY<height;tidY++)
  {
    for(int tidX=0;tidX<width;tidX++)
    {
      float4 summ = m_filterWeights[m_blurRadius]*a_dataIn[pitch(tidX, tidY, m_widthSmall)]; 
     
      for (int wid = 1; wid < m_blurRadius; wid++) //  <--- * --->
      {
        int left  = tidY-wid;
        int right = tidY+wid;
    
        if(left < 0) left = 0;
        if(right >= m_heightSmall) right = m_heightSmall-1;
    
        float4 p0 = m_filterWeights[wid + m_blurRadius]*a_dataIn[pitch(tidX, left, m_widthSmall)]; 
        float4 p1 = m_filterWeights[wid + m_blurRadius]*a_dataIn[pitch(tidX, right, m_widthSmall)]; 
        summ = summ + (p0 + p1);
      }
    
      a_dataOut[pitch(tidX, tidY, m_widthSmall)] = summ;
    }
  }
}

void ToneMapping::kernel2D_MixAndToneMap(int width, int height, const Texture2D<float4>* inData4f, const Sampler* a_sampler, const fTexture2D<float4>* inBrightPixels, unsigned int* outData1ui)
{
  Sampler sampler;
  sampler.m_filter = Sampler::Filter::MIN_MAG_MIP_LINEAR;
  for(int tidY=0;tidY<height;tidY++)
  {
    for(int tidX=0;tidX<width;tidX++)
    {
      const float texCoordX = (float)(tidX) / (float) (m_width);
      const float texCoordY = (float)(tidY) / (float) (m_height);
      float4 sampledColor = sample(sampler, inBrightPixels, int2(m_widthSmall, m_heightSmall), float2(texCoordX, texCoordY) * 0.5);
      float4 colorSumm    = clamp(sampledColor + inData4f[pitch(tidX, tidY, m_width)], 0.0f, 1.0f);
    
      colorSumm.x = pow(colorSumm.x, m_gammaInv);
      colorSumm.y = pow(colorSumm.y, m_gammaInv);
      colorSumm.z = pow(colorSumm.z, m_gammaInv);
      colorSumm.w = 1.0f;
    
      outData1ui[pitch(tidX, tidY, m_width)] = RealColorToUint32( colorSumm );
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMapping::Bloom(int w, int h, const Texture2D<float4>* a_texture2d, const Sampler* a_sampler, unsigned int* outData1ui)
{
  // (1) ExtractBrightPixels (inData4f => m_brightPixels (w,h))
  //
  kernel2D_ExtractBrightPixels(w, h, a_texture2d, a_sampler, &m_brightPixels);

  // (2) Downsample (m_brightPixels => m_downsampledImage (w/4, h/4) )
  //
  kernel2D_DownSample4x(m_widthSmall, m_heightSmall, &m_brightPixels, a_sampler, &m_downsampledImage);

  // (3) GaussBlur (m_downsampledImage => m_downsampledImage)
  //
  kernel2D_BlurX(m_widthSmall, m_heightSmall, &m_downsampledImage, a_sampler, &m_tempImage); // m_downsampledImage => m_tempImage

  kernel2D_BlurY(m_widthSmall, m_heightSmall, &m_tempImage, a_sampler, &m_downsampledImage); // m_tempImage => m_downsampledImage

  // (4) MixAndToneMap(inData4f, m_downsampledImage) => outData1ui
  //
  kernel2D_MixAndToneMap(w,h, a_texture2d, a_sampler, &m_downsampledImage, outData1ui);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void tone_mapping_cpu(int w, int h, float* a_hdrData, const char* a_outName)
{
  ToneMapping filter;  
  Sampler sampler;
  Texture2D<float4> texture(w, h);
  std::vector<uint>  ldrData(w*h);


  filter.SetMaxImageSize(w,h);
  filter.Bloom(w,h, &texture, &sampler, ldrData.data());
  SaveBMP(a_outName, ldrData.data(), w, h);

  // test texture here ... 
  //


  return;
}