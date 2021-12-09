#include "test_class.h"
#include "Bitmap.h"
#include <cassert>

inline static uint pitch(uint x, uint y, uint pitch) { return y*pitch + x; }  


static void SaveTestImage(const float4* data, int w, int h)
{
  std::vector<uint> ldrData(w*h);
  for(size_t i=0;i<w*h;i++)
    ldrData[i] = RealColorToUint32( clamp(data[i], 0.0f, 1.0f));
  SaveBMP("ztest.bmp", ldrData.data(), w, h);
}


static void ConvertSrgbToXyz(float3* a_data)
{
  const float R = a_data->x;
  const float G = a_data->y;
  const float B = a_data->z;

  a_data->x     = R * 0.4124564F + G * 0.3575761F + B * 0.1804375F;
  a_data->y     = R * 0.2126729F + G * 0.7151522F + B * 0.0721750F;
  a_data->z     = R * 0.0193339F + G * 0.1191920F + B * 0.9503041F;
}


static void ConvertXyzToLmsPower(float3* a_data, const float a_power)
{
  float L =  0.4002F * a_data->x + 0.7075F * a_data->y + -0.0807F * a_data->z;
  float M = -0.2280F * a_data->x + 1.1500F * a_data->y +  0.0612F * a_data->z;
  float S =  0.9184F * a_data->z;

  const float a = a_power;

  if      (L >= 0.0F) L =  pow( L, a);
  else if (L <  0.0F) L = -pow(-L, a);
  if      (M >= 0.0F) M =  pow( M, a);
  else if (M <  0.0F) M = -pow(-M, a);
  if      (S >= 0.0F) S =  pow( S, a);
  else if (S <  0.0F) S = -pow(-S, a);

  a_data->x = L;
  a_data->y = M;
  a_data->z = S;
}


static void ConvertLmsToIpt(float3* a_data)
{
  const float I = 0.4000F * a_data->x +  0.4000F * a_data->y +  0.2000F * a_data->z;
  const float P = 4.4550F * a_data->x + -4.8510F * a_data->y +  0.3960F * a_data->z;
  const float T = 0.8056F * a_data->x +  0.3572F * a_data->y + -1.1628F * a_data->z;

  a_data->x     = I;
  a_data->y     = P;
  a_data->z     = T;
}


static void ConvertIptToLms(float3* a_data)
{
  const float L = 0.9999F * a_data->x +  0.0970F * a_data->y +  0.2053F * a_data->z;
  const float M = 0.9999F * a_data->x + -0.1138F * a_data->y +  0.1332F * a_data->z;
  const float S = 0.9999F * a_data->x +  0.0325F * a_data->y + -0.6768F * a_data->z;

  a_data->x     = L;
  a_data->y     = M;
  a_data->z     = S;
}


static void ConvertLmsToXyzPower(float3* a_data, const float a_power)
{
  float L = a_data->x;
  float M = a_data->y;
  float S = a_data->z;  

  if      (L >= 0.0F) L =  pow( L, a_power);
  else if (L <  0.0F) L = -pow(-L, a_power);
  if      (M >= 0.0F) M =  pow( M, a_power);
  else if (M <  0.0F) M = -pow(-M, a_power);
  if      (S >= 0.0F) S =  pow( S, a_power);
  else if (S <  0.0F) S = -pow(-S, a_power);

  a_data->x = 1.8502F * L + -1.1383F * M +  0.2384F * S;
  a_data->y = 0.3668F * L +  0.6438F * M + -0.0106F * S;
  a_data->z = 1.0888F * S;
}


static void ConvertXyzToSrgb(float3* a_data)
{
  const float X = a_data->x;
  const float Y = a_data->y;
  const float Z = a_data->z;

  a_data->x     = X *  3.2404542F + Y * -1.5371385F + Z * -0.4985314F;
  a_data->y     = X * -0.9692660F + Y *  1.8760108F + Z *  0.0415560F;
  a_data->z     = X *  0.0556434F + Y * -0.2040259F + Z *  1.0572252F;
}


static void Blend(float* inData1, const float inData2, const float amount) // 0 - data1, 1 - data2
{
  *inData1 = *inData1 + (inData2 - *inData1) * amount;
}


static void CompressWithKnee(float* a_data, const float a_compress)
{
  float knee = 10.0F;
  Blend(&knee, 2.0F, pow(a_compress, 0.175F)); // lower = softer
  const float antiKnee = 1.0F / knee;  
  
  (*a_data) = (*a_data) / pow((1.0F + pow((*a_data), knee)), antiKnee);
}


static void CompressWithKnee3f(float3* a_data, const float a_compress)
{
  float knee = 10.0F;
  Blend(&knee, 2.0F, pow(a_compress, 0.175F)); // lower = softer
  const float antiKnee = 1.0F / knee;  
  
  a_data->x = a_data->x / pow((1.0F + pow(a_data->x, knee)), antiKnee);
  a_data->y = a_data->y / pow((1.0F + pow(a_data->y, knee)), antiKnee);
  a_data->z = a_data->z / pow((1.0F + pow(a_data->z, knee)), antiKnee);
}


static void MoreCompressColor_IPT(float3* a_dataIPT)
{
  const float saturation = sqrt(a_dataIPT->y * a_dataIPT->y + a_dataIPT->z * a_dataIPT->z);
  const float compSat    = tanh(saturation);
  const float colorDiff  = compSat / fmax(saturation, 1e-6F);

  a_dataIPT->y           *= colorDiff;
  a_dataIPT->z           *= colorDiff;
}


static void CompressIPT(float3* a_dataIPT, const float a_compress)
{
  // Global compress.
  float compLum    = a_dataIPT->x;
  CompressWithKnee(&compLum, a_compress);
  const float diff = pow(compLum / fmax(a_dataIPT->x, 1e-6F), 3.0F - a_compress * 2.0F);

  a_dataIPT->x     = compLum;
  a_dataIPT->y    *= diff;
  a_dataIPT->z    *= diff;
  
  MoreCompressColor_IPT(a_dataIPT);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMapping::SetMaxImageSize(int w, int h)
{
  m_width       = w;
  m_height      = h;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMapping::kernel1D_IPTcompress(int a_size, const float4* inData4f, unsigned int* outData1ui)
{   
  for(int i = 0; i < a_size; i++)
  {      
    float3 pixel = make_float3(inData4f[i].x,inData4f[i].y,inData4f[i].z);

    ConvertSrgbToXyz    (&pixel);
    ConvertXyzToLmsPower(&pixel, 0.43F);
    ConvertLmsToIpt     (&pixel);

    CompressIPT         (&pixel, 1.0F);

    ConvertIptToLms     (&pixel);
    ConvertLmsToXyzPower(&pixel, 1.0F / 0.43F);
    ConvertXyzToSrgb    (&pixel);

    // little compress in RGB
    CompressWithKnee3f(&pixel, 0.01F);

    pixel = clamp(pixel, 0.0F, 1.0F);

    const float4 resColor{
      (float)pow(pixel.x, m_gammaInv), 
      (float)pow(pixel.y, m_gammaInv), 
      (float)pow(pixel.z, m_gammaInv), 1.0f};

    outData1ui[i]         = RealColorToUint32(resColor);    
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ToneMapping::IPTcompress(int w, int h, const float4* inData4f, unsigned int* outData1ui)
{  
  const int size = w*h;
  kernel1D_IPTcompress(size, inData4f, outData1ui); 
}