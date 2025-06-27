#include <vector>
#include <memory>
#include <limits>
#include <cassert>
#include <chrono>
#include <array>

#include "{{IncludeClassDecl}}"

{% for ctorDecl in Constructors %}
{% if ctorDecl.NumParams == 0 %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}(wk_utils::WulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>();
  pObj->InitWulkanObjects(a_ctx.device, a_ctx.physicalDevice, a_maxThreadsGenerated);
  return pObj;
}
{% else %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}({{ctorDecl.Params}}, wk_utils::WulkanContext a_ctx, size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>({{ctorDecl.PrevCall}});
  pObj->InitWulkanObjects(a_ctx.device, a_ctx.physicalDevice, a_maxThreadsGenerated);
  return pObj;
}
{% endif %}
{% endfor %}

static std::string readFile(const char* path) 
{
  std::ifstream file(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

wk_utils::WulkanDeviceFeatures {{MainClassName}}{{MainClassSuffix}}_ListRequiredDeviceFeatures()
{
  wk_utils::WulkanDeviceFeatures res;
  return res;
}

void {{MainClassName}}{{MainClassSuffix}}::InitWulkanObjects(WGPUDevice a_device, WGPUAdapter a_physicalDevice, size_t a_maxThreads)
{
  physicalDevice = a_physicalDevice;
  device         = a_device;
  InitKernels("{{ShaderFolder}}");
}

void {{MainClassName}}{{MainClassSuffix}}::InitKernels(const char* a_path)
{
  {% for Kernel in Kernels %}
  InitKernel_{{Kernel.Name}}(a_path);
  {% endfor %}
}

{% for Kernel in Kernels %}
void {{MainClassName}}{{MainClassSuffix}}::InitKernel_{{Kernel.Name}}(const char* a_filePath)
{
  std::string shaderPath = std::string(a_filePath) + "/{{Kernel.OriginalName}}" + ".wgsl";
  auto shaderSrc = readFile(shaderPath.c_str());

  WGPUShaderModuleWGSLDescriptor wgslDesc = {};
  wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
  wgslDesc.code = shaderSrc.c_str();
  WGPUShaderModuleDescriptor shaderDesc = {};
  shaderDesc.nextInChain = &wgslDesc.chain;
  WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);

  // 5. Create compute pipeline
  WGPUComputePipelineDescriptor pipelineDesc = {};
  pipelineDesc.compute.module     = shaderModule;
  pipelineDesc.compute.entryPoint = "main";

  {{Kernel.Name}}Pipeline = wgpuDeviceCreateComputePipeline(device, &pipelineDesc);
  {{Kernel.Name}}DSLayout = wgpuComputePipelineGetBindGroupLayout({{Kernel.Name}}Pipeline, 0);
}

{% endfor %}

{% for MainFunc in MainFunctions %}
void {{MainClassName}}{{MainClassSuffix}}::UpdateAllBindingGroup_{{MainFunc.Name}}()
{
  // now create actual bindings
  //
  {% for DescriptorSet in MainFunc.DescriptorSets %}
  // descriptor set #{{DescriptorSet.Id}}: {{DescriptorSet.KernelName}}Cmd ({{DescriptorSet.ArgNames}})
  {
    std::array<WGPUBindGroupEntry, {{DescriptorSet.ArgNumber}} + 2> descriptorBufferInfo;
    std::array<WGPUBindGroupEntry, {{DescriptorSet.ArgNumber}} + 2> descriptorImageInfo;

    {% for Arg in DescriptorSet.Args %}
    {% if Arg.IsTexture %}
    //descriptorImageInfo[{{Arg.Id}}].imageView   = {{Arg.Name}}View;
    //descriptorImageInfo[{{Arg.Id}}].imageLayout = {{Arg.AccessLayout}};
    //descriptorImageInfo[{{Arg.Id}}].sampler     = {{Arg.SamplerName}};
    {% else if Arg.IsTextureArray %}
    //std::vector<VkDescriptorImageInfo> {{Arg.NameOriginal}}Info(m_vdata.{{Arg.NameOriginal}}ArrayMaxSize);
    //for(size_t i=0; i<m_vdata.{{Arg.NameOriginal}}ArrayMaxSize; i++)
    //{
    //  if(i < {{Arg.NameOriginal}}.size())
    //  {
    //    {{Arg.NameOriginal}}Info[i].sampler     = m_vdata.{{Arg.NameOriginal}}ArraySampler[i];
    //    {{Arg.NameOriginal}}Info[i].imageView   = m_vdata.{{Arg.NameOriginal}}ArrayView   [i];
    //    {{Arg.NameOriginal}}Info[i].imageLayout = {{Arg.AccessLayout}};
    //  }
    //  else
    //  {
    //    {{Arg.NameOriginal}}Info[i].sampler     = m_vdata.{{Arg.NameOriginal}}ArraySampler[0];
    //    {{Arg.NameOriginal}}Info[i].imageView   = m_vdata.{{Arg.NameOriginal}}ArrayView   [0];
    //    {{Arg.NameOriginal}}Info[i].imageLayout = {{Arg.AccessLayout}};
    //  }
    //}
    {% else if Arg.IsAccelStruct %}
    //{
    //  VulkanRTX* pScene = dynamic_cast<VulkanRTX*>({{Arg.Name}}->UnderlyingImpl(1));
    //  if(pScene == nullptr)
    //    std::cout << "[{{MainClassName}}{{MainClassSuffix}}::InitAllGeneratedDescriptorSets_{{MainFunc.Name}}]: fatal error, wrong accel struct type" << std::endl;
    //  accelStructs       [{{Arg.Id}}] = pScene->GetSceneAccelStruct();
    //  descriptorAccelInfo[{{Arg.Id}}] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,VK_NULL_HANDLE,1,&accelStructs[{{Arg.Id}}]};
    //}
    {% else %}
    descriptorBufferInfo[{{Arg.Id}}]         = WGPUBindGroupEntry{};
    descriptorBufferInfo[{{Arg.Id}}].binding = {{Arg.Id}};
    descriptorBufferInfo[{{Arg.Id}}].buffer  = {{Arg.Name}}Buffer;
    descriptorBufferInfo[{{Arg.Id}}].offset  = {{Arg.Name}}Offset;
    descriptorBufferInfo[{{Arg.Id}}].size    = {{Arg.Name}}Size; 
    {% endif %}
  
    {% endfor %}
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}]         = WGPUBindGroupEntry{};
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].binding = {{DescriptorSet.ArgNumber}};
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].buffer  = m_classDataBuffer;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].offset  = 0;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}].size    = m_classDataSize;
    
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1]         = WGPUBindGroupEntry{};
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].binding = {{DescriptorSet.ArgNumber}}+1;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].buffer  = m_pushConstantBuffer;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].offset  = 0;
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].size    = m_pushConstantSize;

    WGPUBindGroupDescriptor bgDesc = {};
    bgDesc.layout     = {{DescriptorSet.KernelName}}DSLayout; 
    bgDesc.entryCount = descriptorBufferInfo.size();
    bgDesc.entries    = descriptorBufferInfo.data();
    m_allGeneratedDS[{{DescriptorSet.Id}}] = wgpuDeviceCreateBindGroup(device, &bgDesc);
  }
  {% endfor %}
}

{% endfor %}

{% for MainFunc in MainFunctions %}
{{MainFunc.ReturnType}} {{MainClassName}}{{MainClassSuffix}}::{{MainFunc.MainFuncDeclCmd}}
{
  WGPUComputePassDescriptor passDesc = {};
  m_currEncoder = a_commandEncoder;
  m_currPassCS  = wgpuCommandEncoderBeginComputePass(m_currEncoder, &passDesc);
  {% if MainFunc.IsMega %}
  wgpuComputePassEncoderSetBindGroup(m_currPassCS, 0, m_allGeneratedDS[{{MainFunc.DSId}}], 0, nullptr);  
  {{MainFunc.MegaKernelCall}}
  {% else %}
  {{MainFunc.MainFuncTextCmd}}
  {% endif %} {# /* end of else branch */ #}
  wgpuComputePassEncoderEnd(m_currPassCS);
}
{% endfor %}

{{MainClassName}}{{MainClassSuffix}}::~{{MainClassName}}{{MainClassSuffix}}()
{
  
}