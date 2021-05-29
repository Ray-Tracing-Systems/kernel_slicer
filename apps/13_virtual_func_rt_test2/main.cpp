#include <iostream>
#include <fstream>

void test_class_cpu();
void test_class_gpu_V2();
void test_class_gpu_single_queue();

int main(int argc, const char** argv)
{
//  test_class_cpu();
//  test_class_gpu();

  test_class_gpu_single_queue();
  //test_class_gpu_V2();
  return 0;
}
