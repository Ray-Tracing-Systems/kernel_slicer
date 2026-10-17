#pragma once
#include <vector>
#include <cstdint>
#include "optimal_layout.h"

// helper inline functions with uint32_t local variables and references (see optimal_layout.h)
class InlineFuncs
{
public:
  InlineFuncs(){}

  virtual void Run(uint32_t* a_out [[size("a_width*a_height*2")]], uint32_t a_width, uint32_t a_height);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel2D_Layout(uint32_t* a_out, uint32_t a_width, uint32_t a_height);
};
