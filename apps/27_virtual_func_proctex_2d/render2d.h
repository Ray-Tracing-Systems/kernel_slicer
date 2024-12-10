#ifndef TEST_REINHARD_H
#define TEST_REINHARD_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include "LiteMath.h"
using namespace LiteMath;

struct IProcTexture2D
{
  static constexpr uint32_t TAG_EMPTY      = 0;   
  static constexpr uint32_t TAG_COLOR_RED  = 1;
  static constexpr uint32_t TAG_MANDELBROT = 2; 

  virtual uint32_t GetTag() const { return TAG_EMPTY; }      
  virtual float3 Evaluate(float2 tc) const { return float3(0.0f); }

  uint32_t m_tag = TAG_EMPTY;
};

class ProcRender2D 
{
public:

  ProcRender2D();
  ~ProcRender2D();

  virtual void Fractal(int w, int h, uint32_t* outData [[size("w*h")]]);

  virtual void CommitDeviceData() {}                                                           // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; } // will be overriden in generated class    

  static constexpr uint32_t TOTAL_IMPLEMANTATIONS = 2; 

protected:

  virtual void kernel2D_EvaluateTextures(int w, int h, uint32_t* outData);
  float m_time;

  std::vector<IProcTexture2D*> allProcTextures; 
  void InitAllTextures();
};


static inline int mandel(float c_re, float c_im, int count) 
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) 
  {
    if (z_re * z_re + z_im * z_im > 4.)
        break;
    float new_re = z_re*z_re - z_im*z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
      
  }
  return i;
}


struct Red2D : public IProcTexture2D
{
  Red2D() { m_tag = GetTag(); }  

  uint32_t GetTag()          const override { return TAG_COLOR_RED; }      
  float3 Evaluate(float2 tc) const override { return float3(1.0f, 0.0f, 0.0f); }
};

struct Mandelbrot2D : public IProcTexture2D
{
  Mandelbrot2D() { m_tag = GetTag(); }
  uint32_t GetTag() const override { return TAG_MANDELBROT; }      
  
  float3 Evaluate(float2 tc) const override 
  { 
    const int index = mandel(tc.x-0.5f, tc.y, 100);

    const int r1 = std::min((index*128)/32, 255);
    const int g1 = std::min((index*128)/25, 255);
    const int b1 = std::min((index*index), 255);

    const float fr1 = float(r1)/255.0f;
    const float fg1 = float(g1)/255.0f;
    const float fb1 = float(b1)/255.0f;

    return float3(fr1, fg1, fb1); 
  }
};

#endif