#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <cstdint>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Pair
{
  float x;
  float y;
};

// Local pointer variables to member vectors in kernel and in member function: 'const float* p = &m_vec[e];', 'm_vec.data() + e', 'p + k'.
// On GPU such pointer becomes an offset inside buffer: 'p[i]', '*p', 'p->x' access the buffer and 'p' ('p + k') may be passed to functions.
//
class TestLocalGlobalPtrVar
{
public:

  TestLocalGlobalPtrVar(size_t a_size);

  virtual void Run(const int a_size, float* outData [[size("a_size")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:

  void  kernel1D_Run(const int a_size, float* outData);
  float evalNode(uint32_t a_offset, uint32_t a_id); // local pointer variable in member function

  std::vector<float> m_params;
  std::vector<Pair>  m_pairs;
};

#endif
