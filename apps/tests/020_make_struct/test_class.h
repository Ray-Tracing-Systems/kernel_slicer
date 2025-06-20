#pragma once
#include <cstdint>

#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
using LiteMath::float4;
using LiteMath::float3;
using LiteMath::float2;

struct BoxHit
{
  uint32_t id;
  float tHit;
  unsigned id2;
  unsigned id3;
};

struct Cow
{
  float pos;
  float moooo;
  float mass;
  unsigned id2;
};

static inline BoxHit make_BoxHit(uint32_t a_id, float a_t)
{
  BoxHit res;
  res.id   = a_id;
  res.tHit = a_t;
  return res;
}


class TestClass
{
public:
  TestClass(){}

  virtual void Test(BoxHit* a_data [[size("a_size")]], unsigned int a_size);
  void kernel1D_Test(BoxHit* a_data, unsigned int a_size, Cow a_cow);

  uint32_t m_palette[20] = {
    0xffe6194b, 0xff3cb44b, 0xffffe119, 0xff0082c8,
    0xfff58231, 0xff911eb4, 0xff46f0f0, 0xfff032e6,
    0xffd2f53c, 0xfffabebe, 0xff008080, 0xffe6beff,
    0xffaa6e28, 0xfffffac8, 0xff800000, 0xffaaffc3,
    0xff808000, 0xffffd8b1, 0xff000080, 0xff808080
  };

  static constexpr uint32_t m_palette2[20] = {
    0xffe6194b, 0xff3cb44b, 0xffffe119, 0xff0082c8,
    0xfff58231, 0xff911eb4, 0xff46f0f0, 0xfff032e6,
    0xffd2f53c, 0xfffabebe, 0xff008080, 0xffe6beff,
    0xffaa6e28, 0xfffffac8, 0xff800000, 0xffaaffc3,
    0xff808000, 0xffffd8b1, 0xff000080, 0xff808080
  };

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
