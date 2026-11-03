#include "test_class.h"

// Level 3 (leaf): reads the buffer.
//
static inline float c_read_point(const float* a_params, uint32_t a_offset, uint32_t a_id)
{
  return a_params[a_offset + a_id];
}

// Level 2: passes the buffer further.
//
static inline float b_sum_pair(const float* a_params, uint32_t a_offset, uint32_t a_id)
{
  return c_read_point(a_params, a_offset, a_id) + 10.0f*c_read_point(a_params, a_offset, a_id + 1);
}

// Level 1: passes the buffer further ('pointer + k' in nested functions is tested in 076_nested_global_ptr_add).
// Names are 'a_', 'b_', 'c_' and the caller is defined after its callees here, but generated copies
// of these functions may be written in a different order, so GLSL needs their declarations first.
//
static inline float a_top(const float* a_params, uint32_t a_offset, uint32_t a_id)
{
  return b_sum_pair(a_params, a_offset, a_id) + 100.0f*b_sum_pair(a_params, a_offset + 2, a_id);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

TestNestedGlobalPtr::TestNestedGlobalPtr(size_t a_size)
{
  m_params.resize(64);
  for(size_t i=0;i<m_params.size();i++)
    m_params[i] = float(i % 10);
}

void TestNestedGlobalPtr::kernel1D_Run(const int a_size, float* outData)
{
  for(int i=0;i<a_size;i++)
  {
    const uint32_t offset = uint32_t(i % 4)*8;  // offset inside m_params, as for a node of a tree stored in a flat array
    const uint32_t id     = uint32_t(i % 3);

    // Copies of functions for 'm_params' are created only for functions which are called with it from kernel code,
    // so nested functions must be called with 'm_params.data()' here at least once; this code is never executed.
    //
    if(false)
    {
      b_sum_pair  (m_params.data(), offset, id);
      c_read_point(m_params.data(), offset, id);
    }

    outData[i] = a_top(m_params.data(), offset, id) + 1000.0f*a_top(m_params.data(), offset + 1, id);
  }
}

void TestNestedGlobalPtr::Run(const int a_size, float* outData)
{
  kernel1D_Run(a_size, outData);
}
