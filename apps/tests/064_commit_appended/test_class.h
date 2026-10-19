#pragma once
#include <vector>
#include <cstdint>
#include "LiteMath.h"
using LiteMath::uint2;

// Checks CommitAppendedData(): appending to member vectors without full CommitDeviceData().
// Vectors 'm_a' and 'm_b' form group 'geom' ("appendGroups" in kmake.json), 'm_c' is in 'otherGroup',
// which is in "appendRewriteGroups", so 'm_c' is copied entirely by every CommitAppendedData().
// Dump() copies all vectors (and their sizes as seen on GPU) to output arrays, loops are bounded by member vector sizes.
class AppendTest
{
public:
  AppendTest(){}

  virtual void Dump(uint32_t a_maxSize, uint32_t* a_outA [[size("a_maxSize")]], uint2* a_outB [[size("a_maxSize")]],
                                        uint32_t* a_outC [[size("a_maxSize")]], uint32_t* a_outSizes [[size("4")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void CommitAppendedData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_DumpA(uint32_t* a_out);
  void kernel1D_DumpB(uint2* a_out);
  void kernel1D_DumpC(uint32_t* a_out);
  void kernel1D_DumpSizes(uint32_t a_one, uint32_t* a_out);

  std::vector<uint32_t> m_a;
  std::vector<uint2>    m_b;
  std::vector<uint32_t> m_c;
};
