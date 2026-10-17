#include "test_class.h"

void InlineFuncs::kernel2D_Layout(uint32_t* a_out, uint32_t a_width, uint32_t a_height)
{
  for(uint32_t y=0; y<a_height; y++)
  {
    for(uint32_t x=0; x<a_width; x++)
    {
      const uint32_t index = optimal_layout_flatten(x, y);
      uint32_t x2 = 0, y2 = 0;
      optimal_layout_unflatten(index, x2, y2);
      a_out[(y*a_width + x)*2 + 0] = index;
      a_out[(y*a_width + x)*2 + 1] = (x2 == x && y2 == y) ? optimal_layout_layer_size(a_width, a_height) : 0xFFFFFFFFu;
    }
  }
}

void InlineFuncs::Run(uint32_t* a_out, uint32_t a_width, uint32_t a_height)
{
  kernel2D_Layout(a_out, a_width, a_height);
}
