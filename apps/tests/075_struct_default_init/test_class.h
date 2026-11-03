#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <cstdint>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Default member initializers: Slang supports them, GLSL does not, so kslicer removes them for GLSL (with a warning).
// Initializers have comparisons, parentheses, nested braces and comments with '=' to check that only
// the initializers are removed. NOTE: in GLSL variables of such structs are NOT initialized by default,
// so the kernel initializes all fields explicitly.
//
struct Accum
{
  uint32_t count = 0;               // count = number of added values
  float    sum   = 0.0f;
  float    maxV  = -1.0f;           /* maxV = -1 means 'no values' */
  uint32_t flags = (1u << 2) | 1u;
  uint32_t isBig = (3 >= 2) ? 1u : 0u;
};

struct Pair
{
  float    weights[2] = {0.5f, 0.25f};
  uint32_t id         = 7;
};

class TestStructDefaultInit
{
public:

  TestStructDefaultInit(size_t a_size);

  virtual void Run(const int a_size, float* outData [[size("a_size")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:

  void kernel1D_Run(const int a_size, float* outData);
  std::vector<float> m_data;
};

#endif
