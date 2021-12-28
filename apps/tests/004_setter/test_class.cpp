#include "test_class.h"
#include <cstdint>
#include <algorithm>

using std::min;
using std::max;

void ArrayProcess::SetOutput(MyInOut a_out) 
{ 
  m_out = a_out; 
}

void ArrayProcess::ProcessArrays(const int* a_data1, const int* a_data2, unsigned a_dataSize)
{
  kernel1D_ArrayProc(a_data1, a_data2, m_out.someSize);
}

void ArrayProcess::kernel1D_ArrayProc(const int* a_data1, const int* a_data2, unsigned a_dataSize)
{ 
  m_summ    = 0;
  m_minVal  = 10000000*sizeof(m_testArray)/sizeof(m_testArray[0]);
  m_maxVal  = -10000000;

  for(unsigned i=0; i<a_dataSize; i++)
  {
    m_out.summ[i]    = a_data1[i] + a_data2[i];
    m_out.product[i] = a_data1[i] * a_data2[i];

    m_summ += m_out.summ[i];
    m_minVal = min(m_minVal, a_data1[i]);
    m_minVal = min(m_minVal, a_data2[i]);
    m_maxVal = max(m_maxVal, max(a_data1[i], a_data2[i]));

    testData.push_back((float)(m_out.summ[i]));
  }

  m_out.reduction[0] = m_summ;
  m_out.reduction[1] = m_minVal;
  m_out.reduction[2] = m_maxVal;
  int a = 2;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t array_summ_cpu(const std::vector<int32_t>& array)
{
  ArrayProcess performer;
  std::vector<int32_t> summ(array.size()), prod(array.size()), reduct(3);
  
  MyInOut output;
  output.summ    = summ.data();
  output.product = prod.data();
  output.reduction = reduct.data();
  output.someSize  = unsigned(array.size());

  performer.SetOutput(output);
  performer.ProcessArrays(array.data(), array.data(), output.someSize);
  
  std::cout << "summ = " << reduct[0] << std::endl;
  std::cout << "minv = " << reduct[1] << std::endl;
  std::cout << "maxv = " << reduct[2] << std::endl;

  return 0;
}
