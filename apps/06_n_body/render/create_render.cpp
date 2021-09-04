#include "render_common.h"
#include "points_render.h"


std::unique_ptr<IRender> CreateRender(uint32_t w, uint32_t h, RenderEngineType type)
{
  switch(type)
  {
  case RenderEngineType::POINTS_RENDER:
    return std::make_unique<PointsRender>(w, h);

  default:
    return nullptr;
  }
  return nullptr;
}


