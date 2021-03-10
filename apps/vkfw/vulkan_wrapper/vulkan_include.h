#ifndef VULKAN_INCLUDE_H
#define VULKAN_INCLUDE_H

#if defined(__ANDROID__) // Dynamic load, use vulkan_wrapper.h to load vulkan functions
  #include "vulkan_wrapper/vulkan_wrapper.h"
#else
  #include <vulkan/vulkan.h>
#endif

#endif // VULKAN_INCLUDE_H