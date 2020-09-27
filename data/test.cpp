#include <iostream>

namespace n { namespace m { class C {}; } }

class Test
{
public:
  int GetData() const { return m_data; }
private:
  int m_data;
};

int main(int argc, const char** argv) 
{ 
  for(int i=0;i<10;i++)
    std::cout << i << std::endl;

  for(int j=0;j<10;j++)
    std::cout << j << std::endl;

  return 0; 
}
