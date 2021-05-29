#pragma once 

#include "vk_utils.h"
#include "vk_program.h"
#include "vk_copy.h"
#include "vk_buffer.h"

struct IDrawFrameApp
{
  IDrawFrameApp(){}
  virtual ~IDrawFrameApp(){}
  
  virtual void InitWindow() = 0;
  virtual void Init(VkInstance instance, 
                    VkPhysicalDevice a_physicalDevice, 
                    VkDevice a_device) = 0;
  virtual void DoFrame() = 0;
};

std::shared_ptr<IDrawFrameApp> CreateHelloTriImpl();
