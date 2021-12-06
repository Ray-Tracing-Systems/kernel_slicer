#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

int32_t array_summ_cpu(const std::vector<int32_t>& array);
int32_t array_summ_gpu(const std::vector<int32_t>& array);

int main(int argc, const char** argv)
{
  std::vector<int32_t> array(1024);
  for(size_t i=0;i<array.size();i++)
  {
    if(i%3 == 0)
      array[i] = i;
    else
      array[i] = -i;
  }

  auto summ1 = array_summ_cpu(array);
  auto summ2 = array_summ_gpu(array);

  std::cout << "[cpu]: array summ  = " << summ1 << std::endl;
  std::cout << "[gpu]: array summ  = " << summ2 << std::endl;
  
  if(summ1 == summ2)
    std::cout << "test_001 PASSED!" << std::endl;
  else 
    std::cout << "test_001 FAILED!" << std::endl;

  return 0;
}
