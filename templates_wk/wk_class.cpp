#include <vector>
#include <memory>
#include <limits>
#include <cassert>
#include <chrono>
#include <array>

#include "{{IncludeClassDecl}}"

std::shared_ptr<Mandelbrot> Create{{MainClassName}}{{MainClassSuffix}}(wk_utils::WulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  return nullptr;
}

wk_utils::WulkanDeviceFeatures {{MainClassName}}{{MainClassSuffix}}_ListRequiredDeviceFeatures()
{
  vk_utils::WulkanDeviceFeatures res;
  return res;
}