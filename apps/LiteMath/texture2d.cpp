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

