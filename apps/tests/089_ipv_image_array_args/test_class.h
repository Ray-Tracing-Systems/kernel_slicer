#pragma once
#include "LiteMath.h"
#include "Image2d.h"

using LiteMath::float4;
using LiteMath::int2;
using LiteImage::Image2D;

class ImageArrayArgs
{
public:
  ImageArrayArgs() {}
  void SetSize(int a_width, int a_height);
  virtual void Run(int a_width, int a_height, float* a_output [[size("a_width*a_height")]]);
  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {}

protected:
  virtual void kernel2D_Scale(int a_width, int a_height, const Image2D<float4>& a_source, Image2D<float4>& a_target);
  virtual void kernel2D_Read(int a_width, int a_height, const Image2D<float4>& a_first, const Image2D<float4>& a_second, float* a_output);

  Image2D<float4> m_images[4];
};
