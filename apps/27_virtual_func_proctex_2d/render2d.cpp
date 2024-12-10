#include "render2d.h"
#include <algorithm>
#include <chrono>

static inline uint RealColorToUint32_f3(float3 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  uint red = (uint)r, green = (uint)g, blue = (uint)b;
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}

ProcRender2D::ProcRender2D()
{
  InitAllTextures();
}

ProcRender2D::~ProcRender2D()
{
  for(auto& tex : allProcTextures)
    delete tex;
  allProcTextures.clear();
}

void ProcRender2D::InitAllTextures()
{
  allProcTextures.push_back(new Red2D);
  allProcTextures.push_back(new Mandelbrot2D);
  allProcTextures.push_back(new Ocean2D);
}

static inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }

static inline float mi(float2 a){return min(a.x,a.y);}
static inline float ma(float2 a){return max(a.x,a.y);}
static inline float mu(float2 a){return a.x*a.y;}
static inline float ad(float2 a){return a.x+a.y;}
static inline float su(float2 a){return a.x-a.y;}
static inline float sq2(float a){return a*a;}

float CheckerSignMuFract(float2 u){ return sign(mu(.5-fract(u))); }

void ProcRender2D::kernel2D_EvaluateTextures(int w, int h, uint32_t* outData)
{
  #pragma omp parallel for 
  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++) 
    { 
      const float2 texCoord  = float2(float(x)/float(w), float(y)/float(h));
      const float brickColor = CheckerSignMuFract(texCoord*16.0f);
      
      int index = 2; 

      //if(brickColor < 0.5f)
      //{
      //  index = (x + y) % TOTAL_IMPLEMANTATIONS;
      //}
      //else
      //{
      //  index = ((x + y)/16) % (TOTAL_IMPLEMANTATIONS);
      //}

      const float3 color = allProcTextures[index]->Evaluate(texCoord);
      
      outData[y*w+x] = RealColorToUint32_f3(color);

    }
  }
}

void ProcRender2D::Fractal(int w, int h, uint32_t* outData)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_EvaluateTextures(w, h, outData);  
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}