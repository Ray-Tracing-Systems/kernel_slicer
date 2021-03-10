#include "vk_utils.h"
#include "vk_compute_pipeline.h"

#include <iostream>

namespace vkfw 
{ 

#if defined(__ANDROID__)
  void ComputePipelineMaker::CreateShader(AAssetManager* mgr, VkDevice a_device, const std::string& shader_path, const VkSpecializationInfo *specialization, const char* a_mainName)
  {
    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

     // Read the file
    assert(mgr);
    AAsset* file = AAssetManager_open(mgr, shader_path.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    char* fileContent = new char[fileLength];

    AAsset_read(file, fileContent, fileLength);
    AAsset_close(file);

    VkShaderModuleCreateInfo shaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .codeSize = fileLength,
        .pCode = (const uint32_t*)fileContent,
        .flags = 0,
    };

    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(a_device, &shaderModuleCreateInfo, nullptr, &shaderModule);
    assert(result == VK_SUCCESS);
    delete[] fileContent;

    m_allShaders.push_back(shaderModule);

    stage_info.module = shaderModule;
    stage_info.pName  = a_mainName;
    stage_info.pSpecializationInfo = specialization;

    shaderStageInfo = stage_info;
  }
#endif

  void ComputePipelineMaker::CreateShader(VkDevice a_device, const std::string& shader_path, const VkSpecializationInfo* specialization, const char* a_mainName) 
  {
    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;

    auto shaderCode = vk_utils::ReadFile(shader_path.c_str());
    shaderModule = vk_utils::CreateShaderModule(a_device, shaderCode);

    stage_info.module = shaderModule;
    stage_info.pName  = a_mainName;
    stage_info.pSpecializationInfo = specialization;

    shaderStageInfo = stage_info;
  }

  VkPipelineLayout ComputePipelineMaker::MakeLayout(VkDevice a_device, VkDescriptorSetLayout a_dslayout, uint32_t a_pcRangeSize)
  {
    if(m_device != VK_NULL_HANDLE)
    {
      assert(m_device == a_device); // current implemenataion suppose only single device for each maker helper
    }
    m_device = a_device;

    if (a_pcRangeSize)
    {
      pcRange.stageFlags = shaderStageInfo.stage;
      pcRange.offset = 0;
      pcRange.size = a_pcRangeSize;
    }

    pipelineLayoutInfo                        = {};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = a_pcRangeSize ? 1 : 0;
    pipelineLayoutInfo.pPushConstantRanges    = a_pcRangeSize ? &pcRange : nullptr;

    if (a_dslayout != VK_NULL_HANDLE)
    {
      pipelineLayoutInfo.pSetLayouts    = &a_dslayout; //&descriptorSetLayout;
      pipelineLayoutInfo.setLayoutCount = 1;
    }
    else
    {
      pipelineLayoutInfo.pSetLayouts    = VK_NULL_HANDLE;
      pipelineLayoutInfo.setLayoutCount = 0;
    }

    VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));

    m_allLayouts.push_back(m_pipelineLayout);
    return m_pipelineLayout;
  }

  VkPipeline ComputePipelineMaker::MakePipeline(VkDevice a_device)
  {
    pipelineInfo                    = {};
    pipelineInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.flags              = 0;
    pipelineInfo.stage              = shaderStageInfo;
    pipelineInfo.layout             = m_pipelineLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline));
    
    // free all shader modules due to we don't need them anymore
    //
    #ifndef __ANDROID__
    if (shaderModule != VK_NULL_HANDLE)
      vkDestroyShaderModule(a_device, shaderModule, VK_NULL_HANDLE);
    #endif
    
    m_allPipelines.push_back(m_pipeline);
    return m_pipeline;
  }

  void ComputePipelineMaker::DestroyAll()
  {
    if(m_device == VK_NULL_HANDLE)
      return;

    for(auto pipe : m_allPipelines)
      vkDestroyPipeline(m_device, pipe, nullptr);

    for(auto layout : m_allLayouts)
      vkDestroyPipelineLayout(m_device, layout, nullptr);

    #ifdef __ANDROID__
    //for(auto module : m_allShaders)
      //vkDestroyShaderModule(m_device, module, nullptr);
    #endif

    m_device = VK_NULL_HANDLE;
  }

}
