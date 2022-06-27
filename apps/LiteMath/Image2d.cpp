#include "LiteMath.h"
#include "Image2d.h"

#ifdef USE_STB_IMAGE
  #define STB_IMAGE_IMPLEMENTATION
  #include "stb_image.h"
  #define STB_IMAGE_WRITE_IMPLEMENTATION
  #include "stb_image_write.h"
#endif

#include <iostream>
#include <fstream>

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::uint4;
using LiteMath::int4;
using LiteMath::uint3;
using LiteMath::int3;
using LiteMath::int2;
using LiteMath::uint2;
using LiteMath::ushort4;
using LiteMath::uchar4;
using LiteMath::clamp;

//static inline uint pitch(uint x, uint y, uint pitch) { return y * pitch + x; }  

static inline float4 read_array_uchar4(const uchar4* a_data, int offset)
{
  const float mult = 0.003921568f; // (1.0f/255.0f);
  const uchar4 c0  = a_data[offset];
  return mult*float4((float)c0.x, (float)c0.y, (float)c0.z, (float)c0.w);
}

static inline float4 read_array_uchar4(const uint32_t* a_data, int offset)
{
  return read_array_uchar4((const uchar4*)a_data, offset);
}

static inline int4 bilinearOffsets(const float ffx, const float ffy, const LiteImage::Sampler& a_sampler, const int w, const int h)
{
	const int sx = (ffx > 0.0f) ? 1 : -1;
	const int sy = (ffy > 0.0f) ? 1 : -1;

	const int px = (int)(ffx);
	const int py = (int)(ffy);

	int px_w0, px_w1, py_w0, py_w1;

	if (a_sampler.addressU == LiteImage::Sampler::AddressMode::CLAMP)
	{
		px_w0 = (px     >= w) ? w - 1 : px;
		px_w1 = (px + 1 >= w) ? w - 1 : px + 1;

		px_w0 = (px_w0 < 0) ? 0 : px_w0;
		px_w1 = (px_w1 < 0) ? 0 : px_w1;
	}
	else
	{
		px_w0 = px        % w;
		px_w1 = (px + sx) % w;

		px_w0 = (px_w0 < 0) ? px_w0 + w : px_w0;
		px_w1 = (px_w1 < 0) ? px_w1 + w : px_w1;
	}

	if (a_sampler.addressV == LiteImage::Sampler::AddressMode::CLAMP)
	{
		py_w0 = (py     >= h) ? h - 1 : py;
		py_w1 = (py + 1 >= h) ? h - 1 : py + 1;

		py_w0 = (py_w0 < 0) ? 0 : py_w0;
		py_w1 = (py_w1 < 0) ? 0 : py_w1;
	}
	else
	{
		py_w0 = py        % h;
		py_w1 = (py + sy) % h;

		py_w0 = (py_w0 < 0) ? py_w0 + h : py_w0;
		py_w1 = (py_w1 < 0) ? py_w1 + h : py_w1;
	}

	const int offset0 = py_w0*w + px_w0;
	const int offset1 = py_w0*w + px_w1;
	const int offset2 = py_w1*w + px_w0;
	const int offset3 = py_w1*w + px_w1;

	return int4(offset0, offset1, offset2, offset3);
}

///////////////////////////////////////////////////////////////////////

template<> 
float4 LiteImage::Image2D<float4>::sample(const LiteImage::Sampler& a_sampler, float2 a_uv) const
{
  float ffx = a_uv.x * m_fw - 0.5f; // a_texCoord should not be very large, so that the float does not overflow later. 
  float ffy = a_uv.y * m_fh - 0.5f; // This is left to the responsibility of the top level.

  if ((a_sampler.addressU == Sampler::AddressMode::CLAMP) != 0 && ffx < 0) ffx = 0.0f;
  if ((a_sampler.addressV == Sampler::AddressMode::CLAMP) != 0 && ffy < 0) ffy = 0.0f;
  
  float4 res;
  switch (a_sampler.filter)
  {
    case Sampler::Filter::LINEAR:
    {
      // Calculate the weights for each pixel
      //
      const int   px = (int)(ffx);
      const int   py = (int)(ffy);
  
      const float fx  = std::abs(ffx - (float)px);
      const float fy  = std::abs(ffy - (float)py);
      const float fx1 = 1.0f - fx;
      const float fy1 = 1.0f - fy;
  
      const float w1 = fx1 * fy1;
      const float w2 = fx  * fy1;
      const float w3 = fx1 * fy;
      const float w4 = fx  * fy;
  
      const int4 offsets = bilinearOffsets(ffx, ffy, a_sampler, m_width, m_height);
      const float4 f1    = m_data[offsets.x];
      const float4 f2    = m_data[offsets.y];
      const float4 f3    = m_data[offsets.z];
      const float4 f4    = m_data[offsets.w];

      // Calculate the weighted sum of pixels (for each color channel)
      //
      const float outr = f1.x * w1 + f2.x * w2 + f3.x * w3 + f4.x * w4;
      const float outg = f1.y * w1 + f2.y * w2 + f3.y * w3 + f4.y * w4;
      const float outb = f1.z * w1 + f2.z * w2 + f3.z * w3 + f4.z * w4;
      const float outa = f1.w * w1 + f2.w * w2 + f3.w * w3 + f4.w * w4;
  
      res = float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= int(m_width)) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= int(m_height)) ? m_height - 1 : py;
        py = (py < 0) ? 0 : py;
      }
      else
      {
        py = py % m_height;
        py = (py < 0) ? py + m_height : py;
      }
      res = m_data[py*m_width + px];
    }
    break;
  };
  
  return res;
}

// https://www.shadertoy.com/view/WlG3zG
inline float4 exp2m1(float4 v) { return float4(std::exp2(v.x), std::exp2(v.y), std::exp2(v.z), std::exp2(v.w)) - float4(1.0f); }
inline float4 pow_22(float4 x) { return (exp2m1(0.718151f*x)-0.503456f*x)*7.07342f; }

//inline float4 pow_22(float4 x) { x*x*(float4(0.75f) + 0.25f*x); }

template<> 
float4 LiteImage::Image2D<uchar4>::sample(const LiteImage::Sampler& a_sampler, float2 a_uv) const
{
  float ffx = a_uv.x * m_fw - 0.5f; // a_texCoord should not be very large, so that the float does not overflow later. 
  float ffy = a_uv.y * m_fh - 0.5f; // This is left to the responsibility of the top level.

  if ((a_sampler.addressU == Sampler::AddressMode::CLAMP) != 0 && ffx < 0) ffx = 0.0f;
  if ((a_sampler.addressV == Sampler::AddressMode::CLAMP) != 0 && ffy < 0) ffy = 0.0f;
  
  float4 res;
  switch (a_sampler.filter)
  {
    case Sampler::Filter::LINEAR:
    {
      // Calculate the weights for each pixel
      //
      const int   px = (int)(ffx);
      const int   py = (int)(ffy);
  
      const float fx  = std::abs(ffx - (float)px);
      const float fy  = std::abs(ffy - (float)py);
      const float fx1 = 1.0f - fx;
      const float fy1 = 1.0f - fy;
  
      const float w1 = fx1 * fy1;
      const float w2 = fx  * fy1;
      const float w3 = fx1 * fy;
      const float w4 = fx  * fy;
  
      const int4 offsets = bilinearOffsets(ffx, ffy, a_sampler, m_width, m_height);
      const float4 f1    = read_array_uchar4(m_data.data(), offsets.x);
      const float4 f2    = read_array_uchar4(m_data.data(), offsets.y);
      const float4 f3    = read_array_uchar4(m_data.data(), offsets.z);
      const float4 f4    = read_array_uchar4(m_data.data(), offsets.w);

      // Calculate the weighted sum of pixels (for each color channel)
      //
      const float outr = f1.x * w1 + f2.x * w2 + f3.x * w3 + f4.x * w4;
      const float outg = f1.y * w1 + f2.y * w2 + f3.y * w3 + f4.y * w4;
      const float outb = f1.z * w1 + f2.z * w2 + f3.z * w3 + f4.z * w4;
      const float outa = f1.w * w1 + f2.w * w2 + f3.w * w3 + f4.w * w4;
  
      res = float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= int(m_width)) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= int(m_height)) ? m_height - 1 : py;
        py = (py < 0) ? 0 : py;
      }
      else
      {
        py = py % m_height;
        py = (py < 0) ? py + m_height : py;
      }
      res = read_array_uchar4(m_data.data(), py*m_width + px);
    }
    break;
  };

  if(m_srgb)
    res = pow_22(res);
  
  return res;
}


template<> 
float4 LiteImage::Image2D<uint32_t>::sample(const LiteImage::Sampler& a_sampler, float2 a_uv) const
{
  float ffx = a_uv.x * m_fw - 0.5f; // a_texCoord should not be very large, so that the float does not overflow later. 
  float ffy = a_uv.y * m_fh - 0.5f; // This is left to the responsibility of the top level.

  if ((a_sampler.addressU == Sampler::AddressMode::CLAMP) != 0 && ffx < 0) ffx = 0.0f;
  if ((a_sampler.addressV == Sampler::AddressMode::CLAMP) != 0 && ffy < 0) ffy = 0.0f;
  
  float4 res;
  switch (a_sampler.filter)
  {
    case Sampler::Filter::LINEAR:
    {
      // Calculate the weights for each pixel
      //
      const int   px = (int)(ffx);
      const int   py = (int)(ffy);
  
      const float fx  = std::abs(ffx - (float)px);
      const float fy  = std::abs(ffy - (float)py);
      const float fx1 = 1.0f - fx;
      const float fy1 = 1.0f - fy;
  
      const float w1 = fx1 * fy1;
      const float w2 = fx  * fy1;
      const float w3 = fx1 * fy;
      const float w4 = fx  * fy;
  
      const int4 offsets = bilinearOffsets(ffx, ffy, a_sampler, m_width, m_height);
      const float4 f1    = read_array_uchar4(m_data.data(), offsets.x);
      const float4 f2    = read_array_uchar4(m_data.data(), offsets.y);
      const float4 f3    = read_array_uchar4(m_data.data(), offsets.z);
      const float4 f4    = read_array_uchar4(m_data.data(), offsets.w);

      // Calculate the weighted sum of pixels (for each color channel)
      //
      const float outr = f1.x * w1 + f2.x * w2 + f3.x * w3 + f4.x * w4;
      const float outg = f1.y * w1 + f2.y * w2 + f3.y * w3 + f4.y * w4;
      const float outb = f1.z * w1 + f2.z * w2 + f3.z * w3 + f4.z * w4;
      const float outa = f1.w * w1 + f2.w * w2 + f3.w * w3 + f4.w * w4;
  
      res = float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= int(m_width)) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= int(m_height)) ? m_height - 1 : py;
        py = (py < 0) ? 0 : py;
      }
      else
      {
        py = py % m_height;
        py = (py < 0) ? py + m_height : py;
      }
      res = read_array_uchar4(m_data.data(), py*m_width + px);
    }
    break;
  };
  
  if(m_srgb)
    res = pow_22(res);

  return res;

  // unfortunately this doesn not works correctly for bilinear sampling .... 

  /*bool useBorderColor = false;
  a_uv = process_coord(a_sampler.addressU, a_uv, &useBorderColor);
    
  if (useBorderColor) {
    return a_sampler.borderColor;
  }

  const float2 textureSize = make_float2(m_width, m_height);
  const float2 scaledUV    = textureSize * a_uv;
  const int2   baseTexel   = make_int2(scaledUV.x, scaledUV.y);
  const int    stride      = m_width;

  const uchar4* data2      = (const uchar4*)m_data.data();

  switch (a_sampler.filter)
  {
  
  case Sampler::Filter::LINEAR:
  {
    const int2 cornerTexel  = make_int2(baseTexel.x < m_width  - 1 ? baseTexel.x + 1 : baseTexel.x,
                                        baseTexel.y < m_height - 1 ? baseTexel.y + 1 : baseTexel.y);

    const int offset0       = pitch(baseTexel.x  , baseTexel.y  , stride);
    const int offset1       = pitch(cornerTexel.x, baseTexel.y  , stride);
    const int offset2       = pitch(baseTexel.x  , cornerTexel.y, stride);
    const int offset3       = pitch(cornerTexel.x, cornerTexel.y, stride);

    const float2 lerpCoefs  = scaledUV - float2(baseTexel.x, baseTexel.y);
    const uchar4 uData1     = lerp(data2[offset0], data2[offset1], lerpCoefs.x);
    const uchar4 uData2     = lerp(data2[offset2], data2[offset3], lerpCoefs.x);
    const float4 line1Color = (1.0f/255.0f)*float4(uData1.x, uData1.y, uData1.z, uData1.w);
    const float4 line2Color = (1.0f/255.0f)*float4(uData2.x, uData2.y, uData2.z, uData2.w);

    return lerp(line1Color, line2Color, lerpCoefs.y);
  }     
  case Sampler::Filter::NEAREST:
  default:
  {
    const uchar4 uData = data2[pitch(baseTexel.x, baseTexel.y, stride)];
    return (1.0f/255.0f)*float4(uData.x, uData.y, uData.z, uData.w);
  }

  //default:
  //  fprintf(stderr, "Unsupported filter is used.");
  //  break;
  }

  return make_float4(0.0F, 0.0F, 0.0F, 0.0F);
  */
}

/*
template<typename Type>
float2 Image2D<Type>::process_coord(const Sampler::AddressMode mode, const float2 coord, bool* use_border_color) const
{ 
  float2 res = coord;

  switch (mode)
  {
    case Sampler::AddressMode::CLAMP: 
      //res = coord;            
      break;
    case Sampler::AddressMode::WRAP:  
    {
      res = make_float2(std::fmod(coord.x, 1.0), std::fmod(coord.y, 1.0));            
      if (res.x < 0.0F) res.x += 1.0F;
      if (res.y < 0.0F) res.y += 1.0F;      
      break;
    }
    case Sampler::AddressMode::MIRROR:
    {
      const float u = static_cast<int>(coord.x) % 2 ? 1.0f - std::fmod(coord.x, 1.0f) : std::fmod(coord.x, 1.0f);
      const float v = static_cast<int>(coord.y) % 2 ? 1.0f - std::fmod(coord.y, 1.0f) : std::fmod(coord.y, 1.0f);
      res = make_float2(u, v);            
      break;
    }
    case Sampler::AddressMode::MIRROR_ONCE:
      res = make_float2(std::abs(coord.x), std::abs(coord.y));
      break;
    case Sampler::AddressMode::BORDER:
      *use_border_color = *use_border_color || coord.x < 0.0f || coord.x > 1.0f || coord.y < 0.0f || coord.y > 1.0f;
      break;      
    default:
      break;
  }

  return clamp(res, 0.0f, 1.0F);
} 
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace myvulkan 
{
  // Provided by VK_VERSION_1_0
  typedef enum VkFormat {
    VK_FORMAT_UNDEFINED = 0,
    VK_FORMAT_R8_UNORM = 9,               // ***
    VK_FORMAT_R8_SNORM = 10,              // ***
    VK_FORMAT_R8_SRGB = 15,               // ***
    VK_FORMAT_R8G8B8A8_UNORM = 37,        // ****
    VK_FORMAT_R8G8B8A8_SRGB = 43,         // ***
    VK_FORMAT_R16_UNORM = 70,             // ***
    VK_FORMAT_R16_SNORM = 71,             // ***
    VK_FORMAT_R16G16B16A16_UNORM = 91,    // ***
    VK_FORMAT_R32_SFLOAT = 100,           // ***
    VK_FORMAT_R32G32_SFLOAT = 103,        // ***
    VK_FORMAT_R32G32B32A32_SFLOAT = 109,  // ***
  } VkFormat;
};

template<> uint32_t LiteImage::GetVulkanFormat<uint32_t>(bool a_gamma22) { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_UNORM); } // SRGB, UNORM 
template<> uint32_t LiteImage::GetVulkanFormat<uchar4>(bool a_gamma22)   { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_UNORM); }

template<> uint32_t LiteImage::GetVulkanFormat<uint64_t>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R16G16B16A16_UNORM); }
template<> uint32_t LiteImage::GetVulkanFormat<ushort4>(bool a_gamma22)  { return uint32_t(myvulkan::VK_FORMAT_R16G16B16A16_UNORM); }

template<> uint32_t LiteImage::GetVulkanFormat<uint16_t>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R16_UNORM); }
template<> uint32_t LiteImage::GetVulkanFormat<uint8_t>(bool a_gamma22)  { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8_UNORM); }

template<> uint32_t LiteImage::GetVulkanFormat<float4>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32G32B32A32_SFLOAT); }
template<> uint32_t LiteImage::GetVulkanFormat<float2>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32G32_SFLOAT); }
template<> uint32_t LiteImage::GetVulkanFormat<float> (bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32_SFLOAT); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Pixel { unsigned char r, g, b; };

static bool WriteBMP(const char* fname, Pixel* a_pixelData, int width, int height)
{
  int paddedsize = (width*height) * sizeof(Pixel);
  unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
  unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};

  bmpfileheader[ 2] = (unsigned char)(paddedsize    );
  bmpfileheader[ 3] = (unsigned char)(paddedsize>> 8);
  bmpfileheader[ 4] = (unsigned char)(paddedsize>>16);
  bmpfileheader[ 5] = (unsigned char)(paddedsize>>24);

  bmpinfoheader[ 4] = (unsigned char)(width    );
  bmpinfoheader[ 5] = (unsigned char)(width>> 8);
  bmpinfoheader[ 6] = (unsigned char)(width>>16);
  bmpinfoheader[ 7] = (unsigned char)(width>>24);
  bmpinfoheader[ 8] = (unsigned char)(height    );
  bmpinfoheader[ 9] = (unsigned char)(height>> 8);
  bmpinfoheader[10] = (unsigned char)(height>>16);
  bmpinfoheader[11] = (unsigned char)(height>>24);

  std::ofstream out(fname, std::ios::out | std::ios::binary);
  if(!out.is_open())
    return false;
  out.write((const char*)bmpfileheader, 14);
  out.write((const char*)bmpinfoheader, 40);
  out.write((const char*)a_pixelData, paddedsize);
  out.flush();
  out.close();
  return true;
}

bool LiteImage::SaveBMP(const char* fname, const unsigned int* pixels, int w, int h)
{
  std::vector<Pixel> pixels2(w*h);

  for (size_t i = 0; i < pixels2.size(); i++)
  {
    Pixel px;
    px.r       = (pixels[i] & 0x00FF0000) >> 16;
    px.g       = (pixels[i] & 0x0000FF00) >> 8;
    px.b       = (pixels[i] & 0x000000FF);
    pixels2[i] = px;
  }

  return WriteBMP(fname, &pixels2[0], w, h);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<unsigned int> LiteImage::LoadBMP(const char* filename, int* pW, int* pH)
{
  FILE* f = fopen(filename, "rb");

  if(f == NULL)
  {
    (*pW) = 0;
    (*pH) = 0;
    return std::vector<unsigned int>();
  }

  unsigned char info[54];
  if(fread(info, sizeof(unsigned char), 54, f) != 54) // read the 54-byte header
  {
    (*pW) = 0;
    (*pH) = 0;
    return std::vector<unsigned int>();
  }

  int width  = *(int*)&info[18];
  int height = *(int*)&info[22];

  int row_padded      = (width*3 + 3) & (~3);
  unsigned char* data = new unsigned char[row_padded];

  std::vector<unsigned int> res(width*height);

  for(int i = 0; i < height; i++)
  {
    auto check = fread(data, sizeof(unsigned char), row_padded, f);
    if(check != row_padded)
      break;
    for(int j = 0; j < width; j++)
      res[i*width+j] = (uint32_t(data[j*3+0]) << 16) | (uint32_t(data[j*3+1]) << 8)  | (uint32_t(data[j*3+2]) << 0);
  }

  fclose(f);
  delete [] data;

  (*pW) = width;
  (*pH) = height;
  return res;
}

static inline int tonemap(float x, float a_gammaInv) 
{ 
  const int colorLDR = int( std::pow(x, a_gammaInv)*255.0f + float(.5f) );
  if(colorLDR < 0)        return 0;
  else if(colorLDR > 255) return 255;
  else                    return colorLDR;
}

static inline unsigned IntColorUint32(int r, int g, int b) { return unsigned(r | (g << 8) | (b << 16) | 0xFF000000); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<float4>(const char* a_fileName, const LiteImage::Image2D<float4>& a_image, float a_gamma) 
{
  const float gammaInv = 1.0f/a_gamma;
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) 
      fout << tonemap(c[0], gammaInv) << " " << tonemap(c[1], gammaInv) << " " << tonemap(c[2], gammaInv) << " ";
    fout.close();
    return true;
  }
  else if(fileExt == ".image4f")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)* 2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1]*4)*sizeof(float));
    fout.close();
    return true;
  }
  
  if(fileExt == ".bmp" || fileExt == ".BMP" || 
     fileExt == ".png" || fileExt == ".PNG" || 
     fileExt == ".jpg" || fileExt == ".JPG")
  {
    const bool doFlip = (fileExt == ".bmp" || fileExt == ".BMP");
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = doFlip ? (a_image.height() - y - 1)*a_image.width() : y*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        auto c = a_image.data()[offset1 + x];
        flipedYData[offset2+x] = IntColorUint32(tonemap(c[0], gammaInv), tonemap(c[1], gammaInv), tonemap(c[2], gammaInv));
      }
    }

    if(fileExt == ".bmp" || fileExt == ".BMP")
      return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
    #ifdef USE_STB_IMAGE
    else if(fileExt == ".png" || fileExt == ".PNG") 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), a_image.width() * 4);
    else if(fileExt == ".jpg" || fileExt == ".JPG") 
      return stbi_write_jpg(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), 100);
    #else
    else if(fileExt == ".png" || fileExt == ".PNG") 
    {
      std::cout << "[SaveImage<float4>]: '.png' via stbimage is DISABLED!" << std::endl;
      return false;
    }
    else if(fileExt == ".jpg" || fileExt == ".JPG") 
    {
      std::cout << "[SaveImage<float4>]: '.jpg' via stbimage is DISABLED!" << std::endl;
      return false;
    }
    #endif
  }
  
  std::cout << "[SaveImage<float4>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<float3>(const char* a_fileName, const LiteImage::Image2D<float3>& a_image, float a_gamma) 
{
  const float gammaInv = 1.0f/a_gamma;
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) 
      fout << tonemap(c[0], gammaInv) << " " << tonemap(c[1], gammaInv) << " " << tonemap(c[2], gammaInv) << " ";
    fout.close();
    return true;
  }
  else if(fileExt == ".image3f")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)* 2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1]*3)*sizeof(float));
    fout.close();
    return true;
  }
  
  if(fileExt == ".bmp" || fileExt == ".BMP" || 
     fileExt == ".png" || fileExt == ".PNG" || 
     fileExt == ".jpg" || fileExt == ".JPG")
  {
    const bool doFlip = (fileExt == ".bmp" || fileExt == ".BMP");
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = doFlip ? (a_image.height() - y - 1)*a_image.width() : y*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        auto c = a_image.data()[offset1 + x];
        flipedYData[offset2+x] = IntColorUint32(tonemap(c[0], gammaInv), tonemap(c[1], gammaInv), tonemap(c[2], gammaInv));
      }
    }

    if(fileExt == ".bmp" || fileExt == ".BMP")
      return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
    #ifdef USE_STB_IMAGE
    else if(fileExt == ".png" || fileExt == ".PNG") 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), a_image.width() * 4);
    else if(fileExt == ".jpg" || fileExt == ".JPG") 
      return stbi_write_jpg(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), 100);
    #else
    else if(fileExt == ".png" || fileExt == ".PNG") 
    {
      std::cout << "[SaveImage<float3>]: '.png' via stbimage is DISABLED!" << std::endl;
      return false;
    }
    else if(fileExt == ".jpg" || fileExt == ".JPG") 
    {
      std::cout << "[SaveImage<float3>]: '.jpg' via stbimage is DISABLED!" << std::endl;
      return false;
    }
    #endif
  }

  std::cout << "[SaveImage<float3>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<float>(const char* a_fileName, const LiteImage::Image2D<float>& a_image, float a_gamma) 
{
  const float gammaInv = 1.0f/a_gamma;
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) {
      auto val = tonemap(c, gammaInv);
      fout << val << " " << val << " " << val << " ";
    }
    fout.close();
    return true;
  }
  else if(fileExt == ".image1f")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)* 2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1])*sizeof(float));
    fout.close();
    return true;
  }
  
  if(fileExt == ".bmp" || fileExt == ".BMP" || 
     fileExt == ".png" || fileExt == ".PNG" || 
     fileExt == ".jpg" || fileExt == ".JPG")
  {
    const bool doFlip = (fileExt == ".bmp" || fileExt == ".BMP");
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = doFlip ? (a_image.height() - y - 1)*a_image.width() : y*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        auto c   = a_image.data()[offset1 + x];
        auto val = tonemap(c, gammaInv);
        flipedYData[offset2+x] = IntColorUint32(val, val, val);
      }
    }

    if(fileExt == ".bmp" || fileExt == ".BMP")
      return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
    #ifdef USE_STB_IMAGE
    else if(fileExt == ".png" || fileExt == ".PNG") 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), a_image.width() * 4);
    else if(fileExt == ".jpg" || fileExt == ".JPG") 
      return stbi_write_jpg(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)flipedYData.data(), 100);
    #else
    else if(fileExt == ".png" || fileExt == ".PNG")
    {
      std::cout << "[SaveImage<float>]: '.png' via stbimage is DISABLED!" << std::endl;
      return false;
    }
    else if(fileExt == ".jpg" || fileExt == ".JPG")
    {
      std::cout << "[SaveImage<float>]: '.jpg' via stbimage is DISABLED!" << std::endl;
      return false;
    }
    #endif
  }

  std::cout << "[SaveImage<float>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<uint32_t>(const char* a_fileName, const LiteImage::Image2D<uint32_t>& a_image, float a_gamma) 
{
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) {
      auto r = (c & 0x000000FF);
      auto g = (c & 0x0000FF00) >> 8;
      auto b = (c & 0x00FF0000) >> 16;
      fout << r << " " << g << " " << b << " ";
    }
    fout.close();
    return true;
  }
  else if(fileExt == ".image1ui" || fileExt == ".image4ub")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)* 2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1])*sizeof(uint32_t));
    fout.close();
    return true;
  }
  else if(fileExt == ".bmp" || fileExt == ".BMP")
  {
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = (a_image.height() - y - 1)*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        auto c   = a_image.data()[offset1 + x];
        flipedYData[offset2+x] = c;
      }
    }

    return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
  }
  else if(fileExt == ".png" || fileExt == ".PNG")
  {
    #ifdef USE_STB_IMAGE 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)a_image.data(), a_image.width() * 4);
    #else
      std::cout << "[SaveImage<uint32_t>]: '.png' via stbimage is DISABLED!" << std::endl;
      return false;
    #endif
  }
  else if(fileExt == ".jpg" || fileExt == ".JPG")
  {
    #ifdef USE_STB_IMAGE 
      return stbi_write_jpg(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)a_image.data(), 100);
    #else
      std::cout << "[SaveImage<uchar4>]: '.jpg' via stbimage is DISABLED!" << std::endl;
      return false;
    #endif
  }

  std::cout << "[SaveImage<uint32_t>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
bool LiteImage::SaveImage<uchar4>(const char* a_fileName, const LiteImage::Image2D<uchar4>& a_image, float a_gamma) 
{
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ofstream fout(a_fileName, std::fstream::out);
    if(!fout.is_open())
      return false;
    fout << "P3" << std::endl << a_image.width() << " " << a_image.height() << " 255" << std::endl;
    for (auto c : a_image.vector()) {
      auto r = c[0];
      auto g = c[1];
      auto b = c[2];
      fout << int(r) << " " << int(g) << " " << int(b) << " ";
    }
    fout.close();
    return true;
  }
  else if(fileExt == ".image1ui" || fileExt == ".image4ub")
  {
    unsigned wh[2] = { a_image.width(), a_image.height() };
    std::ofstream fout(a_fileName, std::fstream::out | std::ios::binary);
    if(!fout.is_open())
      return false;
    fout.write((char*)wh, sizeof(unsigned)*2);
    fout.write((char*)a_image.data(), size_t(wh[0]*wh[1])*sizeof(uint32_t));
    fout.close();
    return true;
  }
  else if(fileExt == ".bmp" || fileExt == ".BMP")
  {
    std::vector<unsigned> flipedYData(a_image.width()*a_image.height());
    for (unsigned y = 0; y < a_image.height(); y++)
    {
      const unsigned offset1 = (a_image.height() - y - 1)*a_image.width(); 
      const unsigned offset2 = y*a_image.width();
      for(unsigned x=0; x<a_image.width(); x++)
      {
        const auto c           = a_image.data()[offset1 + x];
        flipedYData[offset2+x] = IntColorUint32(int(c[0]), int(c[1]), int(c[2]));
      }
    }

    return SaveBMP(a_fileName, flipedYData.data(), a_image.width(), a_image.height());
  }
  else if(fileExt == ".png" || fileExt == ".PNG")
  {
    #ifdef USE_STB_IMAGE 
      return stbi_write_png(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)a_image.data(), a_image.width() * 4);
    #else
      std::cout << "[SaveImage<uchar4>]: '.png' via stbimage is DISABLED!" << std::endl;
      return false;
    #endif
  }
  else if(fileExt == ".jpg" || fileExt == ".JPG")
  {
    #ifdef USE_STB_IMAGE 
      return stbi_write_jpg(a_fileName, a_image.width(), a_image.height(), 4, (unsigned char*)a_image.data(), 100);
    #else
      std::cout << "[SaveImage<uchar4>]: '.jpg' via stbimage is DISABLED!" << std::endl;
      return false;
    #endif
  }

  std::cout << "[SaveImage<uchar4>]: unsupported extension '" << fileExt.c_str() << "'" << std::endl;
  return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
LiteImage::Image2D<float4> LiteImage::LoadImage<float4>(const char* a_fileName, float a_gamma)
{
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));
  
  LiteImage::Image2D<float4> img;

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ifstream fin(a_fileName);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<float4>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
  
    std::string header;
    std::getline(fin, header);
    if(header != "P3")
    {
      std::cout << "[LoadImage<float4>]: bad PPM header in file '" << a_fileName << "' " << std::endl;
      return img;
    }
  
    int wh[3] = {0,0,0};
    fin >> wh[0] >> wh[1] >> wh[2];
    if(wh[0] <= 0 || wh[1] <=0)
    {
      std::cout << "[LoadImage<float4>]: bad PPM resolution in file '" << a_fileName << "' " << std::endl;
      return img;
    }

    if(wh[2] != 255)
      std::cout << "[LoadImage<float4>]: bad PPM maxval = " << wh[2] << " " << std::endl;
    
    img.resize(wh[0], wh[1]);
    const size_t totalSize = size_t(wh[0]*wh[1]);
    const float  invDiv    = 1.0f/float(wh[2]); 
    size_t i=0;
    while(fin.is_open() && i < totalSize)
    {
      int color[3] = {0,0,0};
      fin >> color[0] >> color[1] >> color[2];
      float4 colf(std::pow(float(color[0])*invDiv, a_gamma), 
                  std::pow(float(color[1])*invDiv, a_gamma), 
                  std::pow(float(color[2])*invDiv, a_gamma), 0.0f);
      img.data()[i] = colf;
      i++;
    }
    fin.close();
  }
  else if(fileExt == ".bmp" || fileExt == ".BMP")
  {
    int w=0, h=0;
    std::vector<unsigned int> data = LoadBMP(a_fileName, &w, &h);
    if(w == 0 || h == 0)
    {
      std::cout << "[LoadImage<float4>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }

    img.resize(w,h);
    const float  invDiv = 1.0f/255.0f; 
    for(int y=0;y<h;y++)
    {
      const int offset1 = (h-y-1)*w;
      const int offset2 = y*w;
      for(int x=0;x<w;x++) 
      {
        unsigned c = data[offset2+x];
        unsigned r = (c & 0x000000FF);
        unsigned g = (c & 0x0000FF00) >> 8;
        unsigned b = (c & 0x00FF0000) >> 16;
        float4 colf(std::pow(float(r)*invDiv, a_gamma), 
                    std::pow(float(g)*invDiv, a_gamma), 
                    std::pow(float(b)*invDiv, a_gamma), 0.0f);
        img.data()[offset1+x] = colf;
      }
    }
  }
  else if(fileExt == ".png" || fileExt == ".PNG" || fileExt == ".jpg" || fileExt == ".JPG")
  {
    #ifdef USE_STB_IMAGE
    int width, height, channels;
    unsigned char *imgData = stbi_load(a_fileName, &width, &height, &channels, 0);
    
    if(imgData == NULL) 
    {
      std::cout << "[LoadImage<float3>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    else if(channels < 3)
    {
       std::cout << "[LoadImage<float3>]: bad channels number << '" << channels << "' in file '" << a_fileName << "' " << std::endl;
      return img;
    }

    img.resize(width,height);
    const size_t imSize = size_t(width*height);
    const float  invDiv = 1.0f/255.0f;
    for(size_t i=0;i<imSize;i++)
    {
      unsigned r = imgData[i*channels+0];
      unsigned g = imgData[i*channels+1];
      unsigned b = imgData[i*channels+2];
      float4 colf(std::pow(float(r)*invDiv, a_gamma), 
                  std::pow(float(g)*invDiv, a_gamma), 
                  std::pow(float(b)*invDiv, a_gamma), 0.0f);
      img.data()[i] = colf;
    }

    stbi_image_free(imgData);
    return img;
    #else
    std::cout << "[LoadImage<float3>]: png/jpg support is DISABLED! File: '" << a_fileName << "' " << std::endl;
    return img;
    #endif
  }  
  else if(fileExt == ".image4f")
  {
    unsigned wh[2] = { 0,0};
    std::ifstream fin(a_fileName, std::ios::binary);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<float4>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    fin.read((char*)wh, sizeof(unsigned)* 2);
    img.resize(wh[0], wh[1]);
    fin.read((char*)img.data(), size_t(wh[0]*wh[1]*4)*sizeof(float));
    fin.close();
  }
  else
    std::cout << "[LiteImage::LoadImage<float4>]: unsopported image format '" << fileExt.c_str() << "'" << std::endl;
  
  return img;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
LiteImage::Image2D<float3> LiteImage::LoadImage<float3>(const char* a_fileName, float a_gamma)
{
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));
  
  LiteImage::Image2D<float3> img;

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ifstream fin(a_fileName);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<float3>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
  
    std::string header;
    std::getline(fin, header);
    if(header != "P3")
    {
      std::cout << "[LoadImage<float3>]: bad PPM header in file '" << a_fileName << "' " << std::endl;
      return img;
    }
  
    int wh[3] = {0,0,0};
    fin >> wh[0] >> wh[1] >> wh[2];
    if(wh[0] <= 0 || wh[1] <=0)
    {
      std::cout << "[LoadImage<float3>]: bad PPM resolution in file '" << a_fileName << "' " << std::endl;
      return img;
    }

    if(wh[2] != 255)
      std::cout << "[LoadImage<float3>]: bad PPM maxval = " << wh[2] << " " << std::endl;
    
    img.resize(wh[0], wh[1]);
    const size_t totalSize = size_t(wh[0]*wh[1]);
    const float  invDiv    = 1.0f/float(wh[2]); 
    size_t i=0;
    while(fin.is_open() && i < totalSize)
    {
      int color[3] = {0,0,0};
      fin >> color[0] >> color[1] >> color[2];
      float3 colf(std::pow(float(color[0])*invDiv, a_gamma), 
                  std::pow(float(color[1])*invDiv, a_gamma), 
                  std::pow(float(color[2])*invDiv, a_gamma));
      img.data()[i] = colf;
      i++;
    }
    fin.close();
  }
  else if(fileExt == ".bmp" || fileExt == ".BMP")
  {
    int w=0, h=0;
    std::vector<unsigned int> data = LoadBMP(a_fileName, &w, &h);
    if(w == 0 || h == 0)
    {
      std::cout << "[LoadImage<float3>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }

    img.resize(w,h);
    const float  invDiv = 1.0f/255.0f; 
    for(int y=0;y<h;y++)
    {
      const int offset1 = (h-y-1)*w;
      const int offset2 = y*w;
      for(int x=0;x<w;x++) 
      {
        unsigned c = data[offset2+x];
        unsigned r = (c & 0x000000FF);
        unsigned g = (c & 0x0000FF00) >> 8;
        unsigned b = (c & 0x00FF0000) >> 16;
        float3 colf(std::pow(float(r)*invDiv, a_gamma), 
                    std::pow(float(g)*invDiv, a_gamma), 
                    std::pow(float(b)*invDiv, a_gamma));
        img.data()[offset1+x] = colf;
      }
    }
  } 
  else if(fileExt == ".png" || fileExt == ".PNG" || fileExt == ".jpg" || fileExt == ".JPG")
  {
    #ifdef USE_STB_IMAGE
    int width, height, channels;
    unsigned char *imgData = stbi_load(a_fileName, &width, &height, &channels, 0);
    
    if(imgData == NULL) 
    {
      std::cout << "[LoadImage<float3>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    else if(channels < 3)
    {
       std::cout << "[LoadImage<float3>]: bad channels number << '" << channels << "' in file '" << a_fileName << "' " << std::endl;
      return img;
    }

    img.resize(width,height);
    const size_t imSize = size_t(width*height);
    const float  invDiv = 1.0f/255.0f;
    for(size_t i=0;i<imSize;i++)
    {
      unsigned r = imgData[i*channels+0];
      unsigned g = imgData[i*channels+1];
      unsigned b = imgData[i*channels+2];
      float3 colf(std::pow(float(r)*invDiv, a_gamma), 
                  std::pow(float(g)*invDiv, a_gamma), 
                  std::pow(float(b)*invDiv, a_gamma));
      img.data()[i] = colf;
    }

    stbi_image_free(imgData);
    return img;
    #else
    std::cout << "[LoadImage<float3>]: png/jpg support is DISABLED! File: '" << a_fileName << "' " << std::endl;
    return img;
    #endif
  } 
  else if(fileExt == ".image3f")
  {
    unsigned wh[2] = { 0,0};
    std::ifstream fin(a_fileName, std::fstream::out | std::ios::binary);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<float3>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    fin.read((char*)wh, sizeof(unsigned)* 2);
    img.resize(wh[0], wh[1]);
    fin.read((char*)img.data(), size_t(wh[0]*wh[1]*3)*sizeof(float));
    fin.close();
  }
  else
    std::cout << "[LiteImage::LoadImage<float3>]: unsopported image format '" << fileExt.c_str() << "'" << std::endl;
  
  return img;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
LiteImage::Image2D<float> LiteImage::LoadImage<float>(const char* a_fileName, float a_gamma)
{
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));

  if(fileExt == ".image1f")
  {
    LiteImage::Image2D<float> img;
    unsigned wh[2] = { 0,0};
    std::ifstream fin(a_fileName, std::fstream::out | std::ios::binary);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<float>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    fin.read((char*)wh, sizeof(unsigned)* 2);
    img.resize(wh[0], wh[1]);
    fin.read((char*)img.data(), size_t(wh[0]*wh[1])*sizeof(float));
    fin.close();
    return img;
  }

  LiteImage::Image2D<float3> rgbImage = LiteImage::LoadImage<float3>(a_fileName, a_gamma);
  Image2D<float> result(rgbImage.width(), rgbImage.height());

  size_t imSize = size_t(rgbImage.width()*rgbImage.height());
  for(size_t i=0;i<imSize;i++) {
    float3 color = rgbImage.data()[i];
    result.data()[i] = 0.333334f*(color[0] + color[1] + color[2]);
  }
  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
LiteImage::Image2D<uint32_t> LiteImage::LoadImage<uint32_t>(const char* a_fileName, float a_gamma)
{
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));
  
  LiteImage::Image2D<uint32_t> img;

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ifstream fin(a_fileName);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<uint>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
  
    std::string header;
    std::getline(fin, header);
    if(header != "P3")
    {
      std::cout << "[LoadImage<uint>]: bad PPM header in file '" << a_fileName << "' " << std::endl;
      return img;
    }
  
    int wh[3] = {0,0,0};
    fin >> wh[0] >> wh[1] >> wh[2];
    if(wh[0] <= 0 || wh[1] <=0)
    {
      std::cout << "[LoadImage<uint>]: bad PPM resolution in file '" << a_fileName << "' " << std::endl;
      return img;
    }

    if(wh[2] != 255)
      std::cout << "[LoadImage<uint>]: bad PPM maxval = " << wh[2] << " " << std::endl;
    
    img.resize(wh[0], wh[1]);
    const size_t totalSize = size_t(wh[0]*wh[1]);
    size_t i=0;
    while(fin.is_open() && i < totalSize)
    {
      unsigned color[3] = {0,0,0};
      fin >> color[0] >> color[1] >> color[2];
      img.data()[i] = color[0] | (color[1] << 8) | (color[2] << 16);
      i++;
    }
    fin.close();
  } 
  else if(fileExt == ".image1ui" || fileExt == ".image4ub")
  {
    unsigned wh[2] = { 0,0};
    std::ifstream fin(a_fileName, std::ios::binary);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<uint>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    fin.read((char*)wh, sizeof(unsigned)* 2);
    img.resize(wh[0], wh[1]);
    fin.read((char*)img.data(), size_t(wh[0]*wh[1])*sizeof(uint32_t));
    fin.close();
  }
  else if(fileExt == ".bmp" || fileExt == ".BMP")
  {
    int w=0, h=0;
    std::vector<unsigned int> data = LoadBMP(a_fileName, &w, &h);
    if(w == 0 || h == 0)
    {
      std::cout << "[LoadImage<uint>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }

    img.resize(w,h);
    for(int y=0;y<h;y++)
    {
      const int offset1 = (h-y-1)*w;
      const int offset2 = y*w;
      for(int x=0;x<w;x++) 
        img.data()[offset1+x] = data[offset2+x];
    }
  }
  else if(fileExt == ".png" || fileExt == ".PNG" || fileExt == ".jpg" || fileExt == ".JPG")
  {
    #ifdef USE_STB_IMAGE
    int width, height, channels;
    unsigned char *imgData = stbi_load(a_fileName, &width, &height, &channels, 0);
    
    if(imgData == NULL) 
    {
      std::cout << "[LoadImage<uint>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    else if(channels < 3)
    {
      std::cout << "[LoadImage<uint>]: bad channels number << '" << channels << "' in file '" << a_fileName << "' " << std::endl;
      return img;
    }

    img.resize(width,height);
    const size_t imSize = size_t(width*height);
    for(size_t i=0;i<imSize;i++)
    {
      unsigned r = imgData[i*channels+0];
      unsigned g = imgData[i*channels+1];
      unsigned b = imgData[i*channels+2];
      img.data()[i] = r | (g << 8) | (b << 16);
    }

    stbi_image_free(imgData);
    return img;
    #else
    std::cout << "[LoadImage<uint>]: png/jpg support is DISABLED! File: '" << a_fileName << "' " << std::endl;
    return img;
    #endif
  }  
  else
    std::cout << "[LiteImage::LoadImage<uint>]: unsopported image format '" << fileExt.c_str() << "'" << std::endl;
  
  img.setSRGB(true);
  return img;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> 
LiteImage::Image2D<uchar4> LiteImage::LoadImage<uchar4>(const char* a_fileName, float a_gamma)
{
  const std::string fileStr(a_fileName);
  const std::string fileExt = fileStr.substr(fileStr.find_last_of('.'));
  
  LiteImage::Image2D<uchar4> img;

  if(fileExt == ".ppm" || fileExt == ".PPM")
  {
    std::ifstream fin(a_fileName);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<uchar4>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
  
    std::string header;
    std::getline(fin, header);
    if(header != "P3")
    {
      std::cout << "[LoadImage<uchar4>]: bad PPM header in file '" << a_fileName << "' " << std::endl;
      return img;
    }
  
    int wh[3] = {0,0,0};
    fin >> wh[0] >> wh[1] >> wh[2];
    if(wh[0] <= 0 || wh[1] <=0)
    {
      std::cout << "[LoadImage<uchar4>]: bad PPM resolution in file '" << a_fileName << "' " << std::endl;
      return img;
    }

    if(wh[2] != 255)
      std::cout << "[LoadImage<uchar4>]: bad PPM maxval = " << wh[2] << " " << std::endl;
    
    img.resize(wh[0], wh[1]);
    const size_t totalSize = size_t(wh[0]*wh[1]);
    size_t i=0;
    while(fin.is_open() && i < totalSize)
    {
      unsigned color[3] = {0,0,0};
      fin >> color[0] >> color[1] >> color[2];
      img.data()[i] = uchar4( (unsigned char)color[0], (unsigned char)color[1], (unsigned char)color[2], 0);
      i++;
    }
    fin.close();
  } 
  else if(fileExt == ".image1ui" || fileExt == ".image4ub")
  {
    unsigned wh[2] = { 0,0};
    std::ifstream fin(a_fileName, std::ios::binary);
    if(!fin.is_open())
    {
      std::cout << "[LoadImage<uchar4>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    fin.read((char*)wh, sizeof(unsigned)* 2);
    img.resize(wh[0], wh[1]);
    fin.read((char*)img.data(), size_t(wh[0]*wh[1])*sizeof(uint32_t));
    fin.close();
  }
  else if(fileExt == ".bmp" || fileExt == ".BMP")
  {
    int w=0, h=0;
    std::vector<unsigned int> data = LoadBMP(a_fileName, &w, &h);
    if(w == 0 || h == 0)
    {
      std::cout << "[LoadImage<uchar4>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }

    img.resize(w,h);
    for(int y=0;y<h;y++)
    {
      const int offset1 = (h-y-1)*w;
      const int offset2 = y*w;
      memcpy((void*)(img.data() + offset1), (void*)(data.data() + offset2), w*sizeof(unsigned));
    }
  }
  else if(fileExt == ".png" || fileExt == ".PNG" || fileExt == ".jpg" || fileExt == ".JPG")
  {
    #ifdef USE_STB_IMAGE
    int width, height, channels;
    unsigned char *imgData = stbi_load(a_fileName, &width, &height, &channels, 0);
    
    if(imgData == NULL) 
    {
      std::cout << "[LoadImage<uchar4>]: can't open file '" << a_fileName << "' " << std::endl;
      return img;
    }
    else if(channels < 3)
    {
      std::cout << "[LoadImage<uchar4>]: bad channels number << '" << channels << "' in file '" << a_fileName << "' " << std::endl;
      return img;
    }

    img.resize(width,height);
    const size_t imSize = size_t(width*height);
    for(size_t i=0;i<imSize;i++)
      img.data()[i] = uchar4(imgData[i*channels+0], imgData[i*channels+1], imgData[i*channels+2], 0);

    stbi_image_free(imgData);
    return img;
    #else
    std::cout << "[LoadImage<uchar4>]: png/jpg support is DISABLED! File: '" << a_fileName << "' " << std::endl;
    return img;
    #endif
  }  
  else
    std::cout << "[LiteImage::LoadImage<uchar4>]: unsopported image format '" << fileExt.c_str() << "'" << std::endl;
  
  img.setSRGB(true);
  return img;
}
