#include "test_class.h"
#include <cstdint>

#include "glm/vec2.hpp" // glm::vec3
#include "glm/vec3.hpp" // glm::vec3
#include "glm/vec4.hpp" // glm::vec4
#include "glm/mat4x4.hpp" // glm::mat4
#include "glm/ext/matrix_transform.hpp" // glm::translate, glm::rotate, glm::scale

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;


void TestClass::Test(BoxHit* a_data, unsigned int a_size)
{
  Cow testCow;
  testCow.moooo = 2.0f;
  kernel1D_Test(a_data, a_size, testCow);
}

void TestClass::kernel1D_Test(BoxHit* a_data, uint32_t a_size, Cow a_cow)
{
  for(uint32_t i=0; i<a_size; i++) 
  {
    vec3 test1 = vec3(1,2,3);
    vec3 test2 = {1.0f,2.0f,3.0f};  
    vec3 test3 = test1 + test2;
    
    mat4 mRot  = mat4(1.0f); //glm::rotate(test1, 3.14159265358979323846f/2.0f, vec3(-1.0f, 0.0f, 0.0f));
    vec4 test4 = mRot*vec4(test2, 1.0f);
    
    a_data[i] = make_BoxHit(i, a_cow.moooo + test4.z);
  }
}
