#include "texture2d.h"

static inline uint pitch(uint x, uint y, uint pitch) { return y * pitch + x; }  

static inline float4 read_array_uchar4(const uchar4* a_data, int offset)
{
  const float mult = 0.003921568f; // (1.0f/255.0f);
  const uchar4 c0  = a_data[offset];
  return mult*make_float4((float)c0.x, (float)c0.y, (float)c0.z, (float)c0.w);
}

static inline float4 read_array_uchar4(const uint32_t* a_data, int offset)
{
  return read_array_uchar4((const uchar4*)a_data, offset);
}

static inline int4 bilinearOffsets(const float ffx, const float ffy, const Sampler& a_sampler, const int w, const int h)
{
	const int sx = (ffx > 0.0f) ? 1 : -1;
	const int sy = (ffy > 0.0f) ? 1 : -1;

	const int px = (int)(ffx);
	const int py = (int)(ffy);

	int px_w0, px_w1, py_w0, py_w1;

	if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
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

	if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
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

	return make_int4(offset0, offset1, offset2, offset3);
}

///////////////////////////////////////////////////////////////////////

template<> 
float4 Texture2D<float4>::sample(const Sampler& a_sampler, float2 a_uv) const
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
  
      res = make_float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= m_width) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= m_height) ? m_height - 1 : py;
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
float4 Texture2D<uchar4>::sample(const Sampler& a_sampler, float2 a_uv) const
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
  
      res = make_float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= m_width) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= m_height) ? m_height - 1 : py;
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
float4 Texture2D<uint32_t>::sample(const Sampler& a_sampler, float2 a_uv) const
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
  
      res = make_float4(outr, outg, outb, outa);
    }
    break;

    case Sampler::Filter::NEAREST:
    default:
    {
      int px = (int)(ffx + 0.5f);
      int py = (int)(ffy + 0.5f);

      if (a_sampler.addressU == Sampler::AddressMode::CLAMP)
      {
        px = (px >= m_width) ? m_width - 1 : px;
        px = (px < 0) ? 0 : px;
      }
      else
      {
        px = px % m_width;
        px = (px < 0) ? px + m_width : px;
      }
  
      if (a_sampler.addressV == Sampler::AddressMode::CLAMP)
      {
        py = (py >= m_height) ? m_height - 1 : py;
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


template<typename Type>
float2 Texture2D<Type>::process_coord(const Sampler::AddressMode mode, const float2 coord, bool* use_border_color) const
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



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace myvulkan {

// Provided by VK_VERSION_1_0
typedef enum VkFormat {
    VK_FORMAT_UNDEFINED = 0,
    VK_FORMAT_R4G4_UNORM_PACK8 = 1,
    VK_FORMAT_R4G4B4A4_UNORM_PACK16 = 2,
    VK_FORMAT_B4G4R4A4_UNORM_PACK16 = 3,
    VK_FORMAT_R5G6B5_UNORM_PACK16 = 4,
    VK_FORMAT_B5G6R5_UNORM_PACK16 = 5,
    VK_FORMAT_R5G5B5A1_UNORM_PACK16 = 6,
    VK_FORMAT_B5G5R5A1_UNORM_PACK16 = 7,
    VK_FORMAT_A1R5G5B5_UNORM_PACK16 = 8,
    VK_FORMAT_R8_UNORM = 9,
    VK_FORMAT_R8_SNORM = 10,
    VK_FORMAT_R8_USCALED = 11,
    VK_FORMAT_R8_SSCALED = 12,
    VK_FORMAT_R8_UINT = 13,
    VK_FORMAT_R8_SINT = 14,
    VK_FORMAT_R8_SRGB = 15,
    VK_FORMAT_R8G8_UNORM = 16,
    VK_FORMAT_R8G8_SNORM = 17,
    VK_FORMAT_R8G8_USCALED = 18,
    VK_FORMAT_R8G8_SSCALED = 19,
    VK_FORMAT_R8G8_UINT = 20,
    VK_FORMAT_R8G8_SINT = 21,
    VK_FORMAT_R8G8_SRGB = 22,
    VK_FORMAT_R8G8B8_UNORM = 23,
    VK_FORMAT_R8G8B8_SNORM = 24,
    VK_FORMAT_R8G8B8_USCALED = 25,
    VK_FORMAT_R8G8B8_SSCALED = 26,
    VK_FORMAT_R8G8B8_UINT = 27,
    VK_FORMAT_R8G8B8_SINT = 28,
    VK_FORMAT_R8G8B8_SRGB = 29,
    VK_FORMAT_B8G8R8_UNORM = 30,
    VK_FORMAT_B8G8R8_SNORM = 31,
    VK_FORMAT_B8G8R8_USCALED = 32,
    VK_FORMAT_B8G8R8_SSCALED = 33,
    VK_FORMAT_B8G8R8_UINT = 34,
    VK_FORMAT_B8G8R8_SINT = 35,
    VK_FORMAT_B8G8R8_SRGB = 36,
    VK_FORMAT_R8G8B8A8_UNORM = 37,
    VK_FORMAT_R8G8B8A8_SNORM = 38,
    VK_FORMAT_R8G8B8A8_USCALED = 39,
    VK_FORMAT_R8G8B8A8_SSCALED = 40,
    VK_FORMAT_R8G8B8A8_UINT = 41,
    VK_FORMAT_R8G8B8A8_SINT = 42,
    VK_FORMAT_R8G8B8A8_SRGB = 43,
    VK_FORMAT_B8G8R8A8_UNORM = 44,
    VK_FORMAT_B8G8R8A8_SNORM = 45,
    VK_FORMAT_B8G8R8A8_USCALED = 46,
    VK_FORMAT_B8G8R8A8_SSCALED = 47,
    VK_FORMAT_B8G8R8A8_UINT = 48,
    VK_FORMAT_B8G8R8A8_SINT = 49,
    VK_FORMAT_B8G8R8A8_SRGB = 50,
    VK_FORMAT_A8B8G8R8_UNORM_PACK32 = 51,
    VK_FORMAT_A8B8G8R8_SNORM_PACK32 = 52,
    VK_FORMAT_A8B8G8R8_USCALED_PACK32 = 53,
    VK_FORMAT_A8B8G8R8_SSCALED_PACK32 = 54,
    VK_FORMAT_A8B8G8R8_UINT_PACK32 = 55,
    VK_FORMAT_A8B8G8R8_SINT_PACK32 = 56,
    VK_FORMAT_A8B8G8R8_SRGB_PACK32 = 57,
    VK_FORMAT_A2R10G10B10_UNORM_PACK32 = 58,
    VK_FORMAT_A2R10G10B10_SNORM_PACK32 = 59,
    VK_FORMAT_A2R10G10B10_USCALED_PACK32 = 60,
    VK_FORMAT_A2R10G10B10_SSCALED_PACK32 = 61,
    VK_FORMAT_A2R10G10B10_UINT_PACK32 = 62,
    VK_FORMAT_A2R10G10B10_SINT_PACK32 = 63,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32 = 64,
    VK_FORMAT_A2B10G10R10_SNORM_PACK32 = 65,
    VK_FORMAT_A2B10G10R10_USCALED_PACK32 = 66,
    VK_FORMAT_A2B10G10R10_SSCALED_PACK32 = 67,
    VK_FORMAT_A2B10G10R10_UINT_PACK32 = 68,
    VK_FORMAT_A2B10G10R10_SINT_PACK32 = 69,
    VK_FORMAT_R16_UNORM = 70,
    VK_FORMAT_R16_SNORM = 71,
    VK_FORMAT_R16_USCALED = 72,
    VK_FORMAT_R16_SSCALED = 73,
    VK_FORMAT_R16_UINT = 74,
    VK_FORMAT_R16_SINT = 75,
    VK_FORMAT_R16_SFLOAT = 76,
    VK_FORMAT_R16G16_UNORM = 77,
    VK_FORMAT_R16G16_SNORM = 78,
    VK_FORMAT_R16G16_USCALED = 79,
    VK_FORMAT_R16G16_SSCALED = 80,
    VK_FORMAT_R16G16_UINT = 81,
    VK_FORMAT_R16G16_SINT = 82,
    VK_FORMAT_R16G16_SFLOAT = 83,
    VK_FORMAT_R16G16B16_UNORM = 84,
    VK_FORMAT_R16G16B16_SNORM = 85,
    VK_FORMAT_R16G16B16_USCALED = 86,
    VK_FORMAT_R16G16B16_SSCALED = 87,
    VK_FORMAT_R16G16B16_UINT = 88,
    VK_FORMAT_R16G16B16_SINT = 89,
    VK_FORMAT_R16G16B16_SFLOAT = 90,
    VK_FORMAT_R16G16B16A16_UNORM = 91,
    VK_FORMAT_R16G16B16A16_SNORM = 92,
    VK_FORMAT_R16G16B16A16_USCALED = 93,
    VK_FORMAT_R16G16B16A16_SSCALED = 94,
    VK_FORMAT_R16G16B16A16_UINT = 95,
    VK_FORMAT_R16G16B16A16_SINT = 96,
    VK_FORMAT_R16G16B16A16_SFLOAT = 97,
    VK_FORMAT_R32_UINT = 98,
    VK_FORMAT_R32_SINT = 99,
    VK_FORMAT_R32_SFLOAT = 100,
    VK_FORMAT_R32G32_UINT = 101,
    VK_FORMAT_R32G32_SINT = 102,
    VK_FORMAT_R32G32_SFLOAT = 103,
    VK_FORMAT_R32G32B32_UINT = 104,
    VK_FORMAT_R32G32B32_SINT = 105,
    VK_FORMAT_R32G32B32_SFLOAT = 106,
    VK_FORMAT_R32G32B32A32_UINT = 107,
    VK_FORMAT_R32G32B32A32_SINT = 108,
    VK_FORMAT_R32G32B32A32_SFLOAT = 109,
    VK_FORMAT_R64_UINT = 110,
    VK_FORMAT_R64_SINT = 111,
    VK_FORMAT_R64_SFLOAT = 112,
    VK_FORMAT_R64G64_UINT = 113,
    VK_FORMAT_R64G64_SINT = 114,
    VK_FORMAT_R64G64_SFLOAT = 115,
    VK_FORMAT_R64G64B64_UINT = 116,
    VK_FORMAT_R64G64B64_SINT = 117,
    VK_FORMAT_R64G64B64_SFLOAT = 118,
    VK_FORMAT_R64G64B64A64_UINT = 119,
    VK_FORMAT_R64G64B64A64_SINT = 120,
    VK_FORMAT_R64G64B64A64_SFLOAT = 121,
    VK_FORMAT_B10G11R11_UFLOAT_PACK32 = 122,
    VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 = 123,
    VK_FORMAT_D16_UNORM = 124,
    VK_FORMAT_X8_D24_UNORM_PACK32 = 125,
    VK_FORMAT_D32_SFLOAT = 126,
    VK_FORMAT_S8_UINT = 127,
    VK_FORMAT_D16_UNORM_S8_UINT = 128,
    VK_FORMAT_D24_UNORM_S8_UINT = 129,
    VK_FORMAT_D32_SFLOAT_S8_UINT = 130,
    VK_FORMAT_BC1_RGB_UNORM_BLOCK = 131,
    VK_FORMAT_BC1_RGB_SRGB_BLOCK = 132,
    VK_FORMAT_BC1_RGBA_UNORM_BLOCK = 133,
    VK_FORMAT_BC1_RGBA_SRGB_BLOCK = 134,
    VK_FORMAT_BC2_UNORM_BLOCK = 135,
    VK_FORMAT_BC2_SRGB_BLOCK = 136,
    VK_FORMAT_BC3_UNORM_BLOCK = 137,
    VK_FORMAT_BC3_SRGB_BLOCK = 138,
    VK_FORMAT_BC4_UNORM_BLOCK = 139,
    VK_FORMAT_BC4_SNORM_BLOCK = 140,
    VK_FORMAT_BC5_UNORM_BLOCK = 141,
    VK_FORMAT_BC5_SNORM_BLOCK = 142,
    VK_FORMAT_BC6H_UFLOAT_BLOCK = 143,
    VK_FORMAT_BC6H_SFLOAT_BLOCK = 144,
    VK_FORMAT_BC7_UNORM_BLOCK = 145,
    VK_FORMAT_BC7_SRGB_BLOCK = 146,
    VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK = 147,
    VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK = 148,
    VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK = 149,
    VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK = 150,
    VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK = 151,
    VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK = 152,
    VK_FORMAT_EAC_R11_UNORM_BLOCK = 153,
    VK_FORMAT_EAC_R11_SNORM_BLOCK = 154,
    VK_FORMAT_EAC_R11G11_UNORM_BLOCK = 155,
    VK_FORMAT_EAC_R11G11_SNORM_BLOCK = 156,
    VK_FORMAT_ASTC_4x4_UNORM_BLOCK = 157,
    VK_FORMAT_ASTC_4x4_SRGB_BLOCK = 158,
    VK_FORMAT_ASTC_5x4_UNORM_BLOCK = 159,
    VK_FORMAT_ASTC_5x4_SRGB_BLOCK = 160,
    VK_FORMAT_ASTC_5x5_UNORM_BLOCK = 161,
    VK_FORMAT_ASTC_5x5_SRGB_BLOCK = 162,
    VK_FORMAT_ASTC_6x5_UNORM_BLOCK = 163,
    VK_FORMAT_ASTC_6x5_SRGB_BLOCK = 164,
    VK_FORMAT_ASTC_6x6_UNORM_BLOCK = 165,
    VK_FORMAT_ASTC_6x6_SRGB_BLOCK = 166,
    VK_FORMAT_ASTC_8x5_UNORM_BLOCK = 167,
    VK_FORMAT_ASTC_8x5_SRGB_BLOCK = 168,
    VK_FORMAT_ASTC_8x6_UNORM_BLOCK = 169,
    VK_FORMAT_ASTC_8x6_SRGB_BLOCK = 170,
    VK_FORMAT_ASTC_8x8_UNORM_BLOCK = 171,
    VK_FORMAT_ASTC_8x8_SRGB_BLOCK = 172,
    VK_FORMAT_ASTC_10x5_UNORM_BLOCK = 173,
    VK_FORMAT_ASTC_10x5_SRGB_BLOCK = 174,
    VK_FORMAT_ASTC_10x6_UNORM_BLOCK = 175,
    VK_FORMAT_ASTC_10x6_SRGB_BLOCK = 176,
    VK_FORMAT_ASTC_10x8_UNORM_BLOCK = 177,
    VK_FORMAT_ASTC_10x8_SRGB_BLOCK = 178,
    VK_FORMAT_ASTC_10x10_UNORM_BLOCK = 179,
    VK_FORMAT_ASTC_10x10_SRGB_BLOCK = 180,
    VK_FORMAT_ASTC_12x10_UNORM_BLOCK = 181,
    VK_FORMAT_ASTC_12x10_SRGB_BLOCK = 182,
    VK_FORMAT_ASTC_12x12_UNORM_BLOCK = 183,
    VK_FORMAT_ASTC_12x12_SRGB_BLOCK = 184,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G8B8G8R8_422_UNORM = 1000156000,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_B8G8R8G8_422_UNORM = 1000156001,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM = 1000156002,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G8_B8R8_2PLANE_420_UNORM = 1000156003,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM = 1000156004,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G8_B8R8_2PLANE_422_UNORM = 1000156005,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM = 1000156006,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_R10X6_UNORM_PACK16 = 1000156007,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_R10X6G10X6_UNORM_2PACK16 = 1000156008,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16 = 1000156009,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 = 1000156010,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 = 1000156011,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 1000156012,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 1000156013,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 1000156014,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 1000156015,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 1000156016,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_R12X4_UNORM_PACK16 = 1000156017,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_R12X4G12X4_UNORM_2PACK16 = 1000156018,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16 = 1000156019,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 = 1000156020,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 = 1000156021,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 1000156022,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 1000156023,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 1000156024,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 1000156025,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 1000156026,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G16B16G16R16_422_UNORM = 1000156027,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_B16G16R16G16_422_UNORM = 1000156028,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM = 1000156029,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G16_B16R16_2PLANE_420_UNORM = 1000156030,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM = 1000156031,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G16_B16R16_2PLANE_422_UNORM = 1000156032,
  // Provided by VK_VERSION_1_1
    VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM = 1000156033,
  // Provided by VK_IMG_format_pvrtc
    VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG = 1000054000,
  // Provided by VK_IMG_format_pvrtc
    VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG = 1000054001,
  // Provided by VK_IMG_format_pvrtc
    VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG = 1000054002,
  // Provided by VK_IMG_format_pvrtc
    VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG = 1000054003,
  // Provided by VK_IMG_format_pvrtc
    VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG = 1000054004,
  // Provided by VK_IMG_format_pvrtc
    VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG = 1000054005,
  // Provided by VK_IMG_format_pvrtc
    VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG = 1000054006,
  // Provided by VK_IMG_format_pvrtc
    VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG = 1000054007,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT = 1000066000,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT = 1000066001,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT = 1000066002,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT = 1000066003,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT = 1000066004,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT = 1000066005,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT = 1000066006,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT = 1000066007,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT = 1000066008,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT = 1000066009,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT = 1000066010,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT = 1000066011,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT = 1000066012,
  // Provided by VK_EXT_texture_compression_astc_hdr
    VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT = 1000066013,
  // Provided by VK_EXT_ycbcr_2plane_444_formats
    VK_FORMAT_G8_B8R8_2PLANE_444_UNORM_EXT = 1000330000,
  // Provided by VK_EXT_ycbcr_2plane_444_formats
    VK_FORMAT_G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16_EXT = 1000330001,
  // Provided by VK_EXT_ycbcr_2plane_444_formats
    VK_FORMAT_G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16_EXT = 1000330002,
  // Provided by VK_EXT_ycbcr_2plane_444_formats
    VK_FORMAT_G16_B16R16_2PLANE_444_UNORM_EXT = 1000330003,
  // Provided by VK_EXT_4444_formats
    VK_FORMAT_A4R4G4B4_UNORM_PACK16_EXT = 1000340000,
  // Provided by VK_EXT_4444_formats
    VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT = 1000340001,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G8B8G8R8_422_UNORM_KHR = VK_FORMAT_G8B8G8R8_422_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_B8G8R8G8_422_UNORM_KHR = VK_FORMAT_B8G8R8G8_422_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_R10X6_UNORM_PACK16_KHR = VK_FORMAT_R10X6_UNORM_PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR = VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_R12X4_UNORM_PACK16_KHR = VK_FORMAT_R12X4_UNORM_PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR = VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G16B16G16R16_422_UNORM_KHR = VK_FORMAT_G16B16G16R16_422_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_B16G16R16G16_422_UNORM_KHR = VK_FORMAT_B16G16R16G16_422_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
  // Provided by VK_KHR_sampler_ycbcr_conversion
    VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
} VkFormat;

};

template<> uint32_t GetVulkanFormat<uint32_t>(bool a_gamma22) { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_UNORM); } // SRGB, UNORM 
template<> uint32_t GetVulkanFormat<uchar4>(bool a_gamma22)   { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_UNORM); }
//template<> uint32_t GetVulkanFormat<char4>(bool a_gamma22)    { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_SNORM) : uint32_t(myvulkan::VK_FORMAT_R8G8B8A8_UNORM); }

template<> uint32_t GetVulkanFormat<uint64_t>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R16G16B16A16_UNORM); }
template<> uint32_t GetVulkanFormat<ushort4>(bool a_gamma22)  { return uint32_t(myvulkan::VK_FORMAT_R16G16B16A16_UNORM); }
//template<> uint32_t GetVulkanFormat<short4>(bool a_gamma22)   { return uint32_t(myvulkan::VK_FORMAT_R16G16B16A16_SNORM); }

template<> uint32_t GetVulkanFormat<uint16_t>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R16_UNORM); }
template<> uint32_t GetVulkanFormat<uint8_t>(bool a_gamma22)  { return a_gamma22 ? uint32_t(myvulkan::VK_FORMAT_R8_SRGB) : uint32_t(myvulkan::VK_FORMAT_R8_UNORM); }

template<> uint32_t GetVulkanFormat<float4>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32G32B32A32_SFLOAT); }
template<> uint32_t GetVulkanFormat<float2>(bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32G32_SFLOAT); }
template<> uint32_t GetVulkanFormat<float> (bool a_gamma22) { return uint32_t(myvulkan::VK_FORMAT_R32_SFLOAT); }

