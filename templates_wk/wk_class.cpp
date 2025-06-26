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