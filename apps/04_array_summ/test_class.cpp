#include "test_class.h"
#include <string>
#include <chrono>

void Numbers::CalcArraySumm(const int* a_data, unsigned int a_dataSize)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel1D_ArraySumm(a_data, a_dataSize);
  m_executionTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}

void Numbers::kernel1D_ArraySumm(const int* a_data, size_t a_dataSize)
{
  double x = 0.0;
  m_summ = 0;
  #pragma omp parallel for reduction(+:m_summ) // num_threads(4)
  for(int i=0; i<a_dataSize; i++)
  {
    //if(a_data[i] > 0)
    //  m_summ += float(a_data[i])*3.14159f;
    if(a_data[i] > 0)
      m_summ += a_data[i];
  }
  m_summ += 10; // test for loop end
}