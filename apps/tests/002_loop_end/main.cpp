#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "ArgParser.h"

int32_t array_summ_cpu(const int32_t* array, const size_t arraySize);
int32_t array_summ_gpu(const int32_t* array, const size_t arraySize, unsigned int a_preferredDeviceId = 0);

int main(int argc, const char** argv)
{
  std::vector<int32_t> array(1024+1);
  for(size_t i=0;i<array.size();i++)
  {
    if(i%3 == 0)
      array[i] = i;
    else
      array[i] = -i;
  }

  array[1024] = 1000;
  ArgParser args(argc, argv);
  unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);

  auto summ1 = array_summ_cpu(array.data(), 1024);
  auto summ2 = array_summ_gpu(array.data(), 1024, a_preferredDeviceId);

  std::cout << "[cpu]: array summ  = " << summ1 << std::endl;
  std::cout << "[gpu]: array summ  = " << summ2 << std::endl;
  
  int returnCode = 0;

  if(summ1 == summ2)
    std::cout << "test_002 PASSED!" << std::endl;
  else
  {
    std::cout << "test_002 FAILED!" << std::endl;
    returnCode = -1;
  }

  return returnCode;
}
