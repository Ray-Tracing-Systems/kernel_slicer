#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>


#include "test_class.h"

std::shared_ptr<Numbers> CreateNumbers_Generated();

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
  
  bool isGPU = true;
  std::shared_ptr<Numbers> pImpl = nullptr;
  if(isGPU)
    pImpl = CreateNumbers_Generated();
  else
    pImpl = std::make_shared<Numbers>();

  pImpl->CalcArraySumm(array.data(), unsigned(array.size()));

  if(isGPU)
    std::cout << "[gpu]: array summ  = " << pImpl->m_summ << std::endl;
  else
    std::cout << "[cpu]: array summ  = " << pImpl->m_summ << std::endl;
  
  return 0;
}
