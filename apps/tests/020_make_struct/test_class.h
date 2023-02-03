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

  virtual void Test(BoxHit* a_data __attribute__((size("a_size"))), unsigned int a_size);
  void kernel1D_Test(BoxHit* a_data, unsigned int a_size);


  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
