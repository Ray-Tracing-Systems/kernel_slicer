#pragma once

#include <string>
#if defined(__ANDROID__) // Dynamic load, use vulkan_wrapper.h to load vulkan functions
  #include "vulkan_wrapper/vulkan_wrapper.h"
#else
  #include <vulkan/vulkan.h>
#endif

namespace vkfw 
{
  class ComputePipelineMaker 
  {
  public:
    ~ComputePipelineMaker() { DestroyAll(); }

    void CreateShader(VkDevice a_device, const std::string& shader_path, const VkSpecializationInfo *specialization = nullptr, const char* a_mainName = "main");
    #if defined(__ANDROID__)
    void CreateShader(AAssetManager* mgr, VkDevice a_device, const std::string& shader_path, const VkSpecializationInfo *specialization = nullptr, const char* a_mainName = "main");
    #endif

    VkPipelineLayout MakeLayout(VkDevice a_device, VkDescriptorSetLayout a_dslayout, uint32_t a_pcRangeSize);
    VkPipeline       MakePipeline(VkDevice a_device);
    void             DestroyAll(); ///<! you may destroy pipelines yourself or call this function to destroy them all at once

  private:
    VkShaderModule                  shaderModule;
    VkPipelineShaderStageCreateInfo shaderStageInfo;

    VkPipelineLayoutCreateInfo  pipelineLayoutInfo;
    VkPushConstantRange         pcRange;
    VkComputePipelineCreateInfo pipelineInfo;

    VkPipelineLayout m_pipelineLayout;
    VkPipeline       m_pipeline;

    VkDevice                      m_device = VK_NULL_HANDLE;
    std::vector<VkShaderModule>   m_allShaders;
    std::vector<VkPipelineLayout> m_allLayouts;
    std::vector<VkPipeline>       m_allPipelines;

  };
}
