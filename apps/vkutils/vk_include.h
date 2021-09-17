#ifndef VULKAN_INCLUDE_H
#define VULKAN_INCLUDE_H

#if defined(USE_VOLK)
#include "volk.h"
#else
#include <vulkan/vulkan.h>
#endif

#endif // VULKAN_INCLUDE_H