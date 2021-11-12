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
      array[i] = -i;
    else
      array[i] = +i;
  }

  array_summ_cpu(array);
  array_summ_gpu(array);
  
  return 0;
}
