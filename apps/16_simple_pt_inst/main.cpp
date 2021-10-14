#include <iostream>
#include <fstream>

void test_class_cpu(const char* a_scenePath);
void test_class_gpu(const char* a_scenePath);

int main(int argc, const char** argv)
{
  test_class_cpu("/home/frol/PROG/msu-graphics-group/scenes/01_simple_scenes/bunny_cornell.xml"); // bunny_cornell.xml
  test_class_gpu("/home/frol/PROG/msu-graphics-group/scenes/01_simple_scenes/bunny_cornell.xml");
  return 0;
}
