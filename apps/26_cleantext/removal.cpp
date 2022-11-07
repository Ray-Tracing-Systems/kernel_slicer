#include "removal.h"
#include <algorithm>
#include <chrono>

static inline int3 ReadRGB(uint32_t colorPacked)
{
  return make_int3((int32_t)( (colorPacked & 0x000000FF)      ),
                   (int32_t)( (colorPacked & 0x0000FF00) >> 8 ),
                   (int32_t)( (colorPacked & 0x00FF0000) >> 16));
}

static inline int estimateDiff(uint32_t colorPacked)
{
  const int3 color = ReadRGB(colorPacked);
  return std::abs(color.x - color.y) + std::abs(color.x - color.z) + std::abs(color.y - color.z);
}


void TextRemoval::Reserve(int w, int h) 
{ 
  m_badPixels.reserve(w*h); 
  m_badPixels.resize(0); 
  m_mask.resize(w*h); 
  m_width = w; 
  m_height = h; 
}

void TextRemoval::kernel2D_findBadPixels(int w, int h, const uint32_t* inData)
{
  m_badPixels.resize(0);
  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++)
    { 
      const int ws = 2;
      bool found = false;
      for(int y1=std::max(y-ws,0); y1 < std::min(y+ws,h);y1++) {
        for(int x1=std::max(x-ws,0); x1 < std::min(x+ws,w);x1++) {  
          const uint32_t colorPacked = inData[y1*w+x1];
          const int32_t  diff        = estimateDiff(colorPacked);
          if(diff >= 8)
            found = true;
        }
      }
      if(found)
        m_badPixels.push_back((y << 16) | x);
      m_mask[y*w+x] = found ? 1 : 0;
    }
  }
}

void TextRemoval::kernel2D_emplaceBadPixels(int w, int h, const uint32_t* inData, uint32_t* outData)
{
  for(int index = 0; index < m_badPixels.size();index++)
  {
    const int packedIndex = m_badPixels[index];
    const int y = (packedIndex & 0xFFFF0000) >> 16;
    const int x = (packedIndex & 0x0000FFFF);
    
    
    int3 left   = make_int3(0,0,0);
    int3 right  = make_int3(0,0,0);
    int3 top    = make_int3(0,0,0);
    int3 bottom = make_int3(0,0,0);

    int leftCounter = 0;
    int rightCounter = 0;
    int topCounter = 0;
    int bottomCounter = 0;
    
    for(int offset=1; offset < 50; offset++) {

      if(x + offset < w) {
        const int maskVal = m_mask[y*w+x+offset];
        if(maskVal == 0) {
          right += ReadRGB(inData[y*w+x+offset]); 
          rightCounter++;
        }
      }

      if(x - offset >= 0) {
        const int maskVal = m_mask[y*w+x-offset];
        if(maskVal == 0) {
          left += ReadRGB(inData[y*w+x-offset]); 
          leftCounter++;
        }
      }
      
      if(y + offset < h) {
        const int maskVal = m_mask[(y+offset)*w+x];
        if(maskVal == 0) {
          top += ReadRGB(inData[(y+offset)*w+x]); 
          topCounter++;
        }
      }
      
      if(y - offset >= 0) {
        const int maskVal = m_mask[(y-offset)*w+x];
        if(maskVal == 0) {
          bottom += ReadRGB(inData[(y-offset)*w+x]); 
          bottomCounter++;
        }
      }

    }

    const float red = (0.5f*(float)left.x/std::max((float)leftCounter,1.0f) + 0.5f*(float)right.x/std::max((float)rightCounter,1.0f) + \
                         0.25f*(float)top.x/std::max((float)topCounter,1.0f)  + 0.25f*(float)bottom.x/std::max((float)bottomCounter,1.0f))/1.5f;

    const float green = (0.5f*(float)left.y/std::max((float)leftCounter,1.0f) + 0.5f*(float)right.y/std::max((float)rightCounter,1.0f) + \
                         0.25f*(float)top.y/std::max((float)topCounter,1.0f)  + 0.25f*(float)bottom.y/std::max((float)bottomCounter,1.0f))/1.5f;

    const float blue = (0.5f*(float)left.z/std::max((float)leftCounter,1.0f) + 0.5f*(float)right.z/std::max((float)rightCounter,1.0f) + \
                        0.25f*(float)top.z/std::max((float)topCounter,1.0f)  + 0.25f*(float)bottom.z/std::max((float)bottomCounter,1.0f))/1.5f;

    int r1 = (int)( std::min(red,   255.0f)  );
    int g1 = (int)( std::min(green, 255.0f)  );
    int b1 = (int)( std::min(blue,  255.0f)  );
      
    outData[y*w+x] = 0xFF000000 | (r1) | (g1 << 8) | (b1 << 16);
    //outData[y*m_width+x] = 0x000000FF;
  }
}


void TextRemoval::Run(int w, int h, const uint32_t* inData, uint32_t* outData)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_findBadPixels(w,h,inData);
  if(inData != outData)
    memcpy(outData, inData, sizeof(uint32_t)*w*h);
  kernel2D_emplaceBadPixels(w,h,inData,outData);
  m_timeEx = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}