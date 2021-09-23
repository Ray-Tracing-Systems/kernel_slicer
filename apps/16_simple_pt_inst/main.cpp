#include <iostream>
#include <fstream>

void test_class_cpu();
void test_class_gpu();

int main(int argc, const char** argv)
{
  //for (auto i : range(1,10)) 
  //  std::cout << i << " ";
  //std::cout << std::endl;

  //test_class_cpu();
  test_class_gpu();
  return 0;
}
