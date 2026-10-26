#include "test_class.h"
#include <new>

void VirtualShapes::InitShapes()
{
  m_shapes.resize(5);
  new (m_shapes.data() + 0) EmptyShape();
  new (m_shapes.data() + 1) Circle(0.5f, 0.5f, 0.4f);
  new (m_shapes.data() + 2) Square(0.4f, 0.6f, 0.3f);
  new (m_shapes.data() + 3) Ring  (0.5f, 0.5f, 0.3f);
  new (m_shapes.data() + 4) Circle(0.2f, 0.7f, 0.2f);
}

void VirtualShapes::Render(uint32_t a_width, uint32_t a_height, float* a_out)
{
  kernel2D_Render(a_width, a_height, a_out);
}

void VirtualShapes::kernel2D_Render(uint32_t a_width, uint32_t a_height, float* a_out)
{
  for(uint32_t y = 0; y < a_height; y++)
  {
    for(uint32_t x = 0; x < a_width; x++)
    {
      const uint32_t shapeId = ((x/8) + (y/8)) % 5;                  // every tile of 8x8 pixels uses its own shape
      const float2   p       = float2(float(x % 8) + 0.5f, float(y % 8) + 0.5f)*(1.0f/8.0f);
      a_out[y*a_width + x]   = (m_shapes.data() + shapeId)->Eval(p);
    }
  }
}
