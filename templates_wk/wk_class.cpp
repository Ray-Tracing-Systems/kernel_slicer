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
  wk_utils::WulkanDeviceFeatures res = {};
  return res;
}

void {{MainClassName}}{{MainClassSuffix}}::InitWulkanObjects(WGPUDevice a_device, WGPUAdapter a_physicalDevice, size_t a_maxThreads)
{
  physicalDevice = a_physicalDevice;
  device         = a_device;
  queue          = wgpuDeviceGetQueue(device);
  InitKernels("{{ShaderFolder}}");
}

void {{MainClassName}}{{MainClassSuffix}}::InitDeviceData()
{
  WGPUBufferDescriptor uboDesc = {};
  uboDesc.size  = sizeof(m_uboData);
  uboDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;

  WGPUBufferDescriptor pcbDesc = {};
  pcbDesc.size  = m_pushConstantStride*m_totalDSNumber;
  pcbDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

  m_classDataBuffer    = wgpuDeviceCreateBuffer(device, &uboDesc);
  m_pushConstantBuffer = wgpuDeviceCreateBuffer(device, &pcbDesc);
  m_classDataSize      = uboDesc.size;         // total size of ubo buffer
  
  {% if length(ClassVectorVars) != 0 %}
  WGPUBufferDescriptor bufDesc = {};
  {% endif %}
  {% for Var in ClassVectorVars %}
  bufDesc.size  = {{Var.Name}}{{Var.AccessSymb}}capacity()*sizeof({{Var.TypeOfData}});
  bufDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
  m_vdata.{{Var.Name}}Buffer = wgpuDeviceCreateBuffer(device, &bufDesc);
  m_vdata.{{Var.Name}}Size   = bufDesc.size;
  {% endfor %}
}

void {{MainClassName}}{{MainClassSuffix}}::GetExecutionTime(const char* a_funcName, float a_out[4]) // NOT IMPLEMENTED 
{
  a_out[0] = 0.0f;
  a_out[1] = 0.0f;
  a_out[2] = 0.0f;
  a_out[3] = 0.0f;
}

void {{MainClassName}}{{MainClassSuffix}}::InitKernels(const char* a_path)
{
  {% for Kernel in Kernels %}
  InitKernel_{{Kernel.Name}}(a_path);
  {% endfor %}
}

void {{MainClassName}}{{MainClassSuffix}}::UpdatePlainMembers()
{
  const size_t maxAllowedSize = std::numeric_limits<uint32_t>::max();
  {% if HasPrefixData %}
  auto pUnderlyingImpl = dynamic_cast<{{PrefixDataClass}}*>({{PrefixDataName}}->UnderlyingImpl(0));
  {% endif %}
  {% for Var in ClassVars %}
  {% if Var.IsArray %}
  {% if Var.HasPrefix %}
  memcpy(m_uboData.{{Var.Name}}, pUnderlyingImpl->{{Var.CleanName}},sizeof(m_uboData.{{Var.Name}}));
  {% else %}
  memcpy(m_uboData.{{Var.Name}},{{Var.Name}},sizeof({{Var.Name}}));
  {% endif %}
  {% else %}
  {% if Var.HasPrefix %}
  m_uboData.{{Var.Name}} = pUnderlyingImpl->{{Var.CleanName}};
  {% else %}
  m_uboData.{{Var.Name}} = {{Var.Name}};
  {% endif %}
  {% endif %}
  {% endfor %}
  {% for Var in ClassVectorVars %}
  m_uboData.{{Var.Name}}_size     = uint32_t( {{Var.Name}}{{Var.AccessSymb}}size() );     assert( {{Var.Name}}{{Var.AccessSymb}}size() < maxAllowedSize );
  m_uboData.{{Var.Name}}_capacity = uint32_t( {{Var.Name}}{{Var.AccessSymb}}capacity() ); assert( {{Var.Name}}{{Var.AccessSymb}}capacity() < maxAllowedSize );
  {% endfor %}
  m_uboData.dummy_last = 0; // Slang to WebGPU issue: access 'ubo[0].dummy_last' prevent slang compiler to discard ubo if it is not used
  wgpuQueueWriteBuffer(queue, m_classDataBuffer, 0, &m_uboData, sizeof(m_uboData));
}

void {{MainClassName}}{{MainClassSuffix}}::UpdateVectorMembers()
{
  {% for Var in ClassVectorVars %}
  if({{Var.Name}}{{Var.AccessSymb}}size() > 0)
    wgpuQueueWriteBuffer(queue, m_vdata.{{Var.Name}}Buffer, 0, {{Var.Name}}{{Var.AccessSymb}}data(), {{Var.Name}}{{Var.AccessSymb}}size()*sizeof({{Var.TypeOfData}}) );
  {% endfor %}
}

{% for UpdateFun in UpdateVectorFun %}
{% if UpdateFun.NumParams == 2 %}
void {{MainClassName}}{{MainClassSuffix}}::{{UpdateFun.Name}}(size_t a_first, size_t a_size)
{
  if(a_first + a_size > {{UpdateFun.VectorName}}.size())
  {
    std::cout << "[{{MainClassName}}{{MainClassSuffix}}::{{UpdateFun.Name}}]: FAILED! wrong first element or size or both " << std::endl;
    return;
  }

  wgpuQueueWriteBuffer(queue, m_vdata.{{UpdateFun.VectorName}}Buffer, a_first*sizeof({{UpdateFun.TypeOfData}}), {{UpdateFun.VectorName}}.data() + a_first, a_size*sizeof({{UpdateFun.TypeOfData}}) );
}
{%else%}
void {{MainClassName}}{{MainClassSuffix}}::{{UpdateFun.Name}}()
{
  if({{UpdateFun.VectorName}}.size() != 0)
    wgpuQueueWriteBuffer(queue, m_vdata.{{UpdateFun.VectorName}}Buffer, 0, {{UpdateFun.VectorName}}.data(), {{UpdateFun.VectorName}}.size()*sizeof({{UpdateFun.TypeOfData}}) );
}
{%endif%}
{% endfor %}

{% for Kernel in Kernels %}
void {{MainClassName}}{{MainClassSuffix}}::InitKernel_{{Kernel.Name}}(const char* a_filePath)
{
  const std::string shaderPath1 = std::string(a_filePath) + "/{{Kernel.OriginalName}}" + ".wgsl";
  const std::string shaderPath  = AlterShaderPath(shaderPath1.c_str());
  auto shaderSrc = readFile(shaderPath.c_str());

  #if WGPU_DISTR >= 30
  {% if WGPU_VER > 30 %}
  const WGPUChainedStruct          tmp1 = {.sType = WGPUSType_ShaderSourceWGSL };
  const WGPUShaderSourceWGSL       tmp2 = {.chain = tmp1, .code = {shaderSrc.c_str(), WGPU_STRLEN},};
  const WGPUShaderModuleDescriptor tmp3 = {.nextInChain = (const WGPUChainedStruct *)&tmp2, .label = {"{{Kernel.OriginalName}} shader module", WGPU_STRLEN} };
  WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(device, &tmp3);
  {% else %}
  WGPUShaderModuleWGSLDescriptor wgslDesc = {};
  wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgslDesc.code = {shaderSrc.c_str(), WGPU_STRLEN};
  WGPUShaderModuleDescriptor shaderDesc = {};
  shaderDesc.nextInChain = &wgslDesc.chain;
  WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);
  {% endif %}

  WGPUComputePipelineDescriptor pipelineDesc = {};
  pipelineDesc.compute.module     = shaderModule;
  pipelineDesc.compute.entryPoint = {"main", WGPU_STRLEN};
  #else
  WGPUShaderModuleWGSLDescriptor wgslDesc = {};
  wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
  wgslDesc.code = shaderSrc.c_str();
  WGPUShaderModuleDescriptor shaderDesc = {};
  shaderDesc.nextInChain = &wgslDesc.chain;
  WGPUShaderModule shaderModule = wgpuDeviceCreateShaderModule(device, &shaderDesc);
  WGPUComputePipelineDescriptor pipelineDesc = {};
  pipelineDesc.compute.module     = shaderModule;
  pipelineDesc.compute.entryPoint = "main";
  #endif

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
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].offset  = m_pushConstantStride*{{DescriptorSet.Id}}; // offset for {{DescriptorSet.Id}} binding group
    descriptorBufferInfo[{{DescriptorSet.ArgNumber}}+1].size    = m_pushConstantStride;

    WGPUBindGroupDescriptor bgDesc = {};
    bgDesc.layout     = {{DescriptorSet.KernelName}}DSLayout; 
    bgDesc.entryCount = descriptorBufferInfo.size();
    bgDesc.entries    = descriptorBufferInfo.data();
    m_allGeneratedDS[{{DescriptorSet.Id}}] = wgpuDeviceCreateBindGroup(device, &bgDesc);
  }
  {% endfor %}
}

{% endfor %}

{% for Kernel in Kernels %}
{% if Kernel.IsIndirect %}
void {{MainClassName}}{{MainClassSuffix}}::{{Kernel.Name}}_UpdateIndirect()
{
  //VkBufferMemoryBarrier barIndirect = BarrierForIndirectBufferUpdate(m_indirectBuffer);
  //vkCmdBindDescriptorSets(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectUpdateLayout, 0, 1, &m_indirectUpdateDS, 0, nullptr);
  //vkCmdBindPipeline      (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectUpdate{{Kernel.Name}}Pipeline);
  //vkCmdDispatch          (m_currCmdBuffer, 1, 1, 1);
  //vkCmdPipelineBarrier   (m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, nullptr, 1, &barIndirect, 0, nullptr);
}

{% endif %}
void {{MainClassName}}{{MainClassSuffix}}::{{Kernel.Decl}}
{
  uint32_t blockSizeX = {{Kernel.WGSizeX}};
  uint32_t blockSizeY = {{Kernel.WGSizeY}};
  uint32_t blockSizeZ = {{Kernel.WGSizeZ}};

  struct KernelArgsPC
  {
    {% for Arg in Kernel.AuxArgs %}
    {{Arg.Type}} m_{{Arg.Name}};
    {% endfor %}
    uint32_t m_sizeX;
    uint32_t m_sizeY;
    uint32_t m_sizeZ;
  } pcData;

  {% if Kernel.SmplX %}
  uint32_t sizeX  = uint32_t({{Kernel.tidX}});
  {% else %}
  uint32_t sizeX  = uint32_t(std::abs(int32_t({{Kernel.tidX}}) - int32_t({{Kernel.begX}})));
  {% endif %}
  {% if Kernel.SmplY %}
  uint32_t sizeY  = uint32_t({{Kernel.tidY}});
  {% else %}
  uint32_t sizeY  = uint32_t(std::abs(int32_t({{Kernel.tidY}}) - int32_t({{Kernel.begY}})));
  {% endif %}
  {% if Kernel.SmplZ %}
  uint32_t sizeZ  = uint32_t({{Kernel.tidZ}});
  {% else %}
  uint32_t sizeZ  = uint32_t(std::abs(int32_t({{Kernel.tidZ}}) - int32_t({{Kernel.begZ}})));
  {% endif %}

  pcData.m_sizeX  = {{Kernel.tidX}};
  pcData.m_sizeY  = {{Kernel.tidY}};
  pcData.m_sizeZ  = {{Kernel.tidZ}};
  {% for Arg in Kernel.AuxArgs %}
  pcData.m_{{Arg.Name}} = {{Arg.Name}};
  {% endfor %}
  {% if Kernel.HasLoopFinish %}
  KernelArgsPC oldPCData = pcData;
  {% endif %}
  
  wgpuQueueWriteBuffer(queue, m_pushConstantBuffer, m_currPCOffset, &pcData, sizeof(KernelArgsPC)); // push constant emulation
  {% if Kernel.HasLoopInit %}
  //vkCmdDispatch(m_currCmdBuffer, 1, 1, 1); // init kernel
  {% endif %}
  {# /* --------------------------------------------------------------------------------------------------------------------------------------- */ #}
  {% if Kernel.IsIndirect %}
  //vkCmdBindPipeline    (m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Pipeline);
  //vkCmdDispatchIndirect(m_currCmdBuffer, m_indirectBuffer, {{Kernel.IndirectOffset}}*sizeof(uint32_t)*4);
  {% else %}
  wgpuComputePassEncoderSetPipeline(m_currPassCS, {{Kernel.Name}}Pipeline);
  wgpuComputePassEncoderDispatchWorkgroups(m_currPassCS, (sizeX + blockSizeX - 1) / blockSizeX, (sizeY + blockSizeY - 1) / blockSizeY, (sizeZ + blockSizeZ - 1) / blockSizeZ);
  {% endif %} {# /* NOT INDIRECT DISPATCH */ #}
  {# /* --------------------------------------------------------------------------------------------------------------------------------------- */ #}
  {% if Kernel.HasLoopFinish %}
  //vkCmdDispatch(m_currCmdBuffer, 1, 1, 1); // finish kernel
  {% endif %}
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

void {{MainClassName}}{{MainClassSuffix}}::ReadBufferBack(WGPUBuffer a_buffer, size_t a_size, void* a_data)
{
  wk_utils::WulkanContext ctx;
  ctx.device         = device;
  ctx.physicalDevice = physicalDevice;
  ctx.instance       = nullptr; //TODO: set instance 
  wk_utils::readBufferBack(ctx, queue, a_buffer, m_readBackBuffer, a_size, a_data);
}

{% for MainFunc in MainFunctions %}
{% if MainFunc.OverrideMe %}

{{MainFunc.ReturnType}} {{MainClassName}}{{MainClassSuffix}}::{{MainFunc.DeclOrig}}
{
  // (1) reserved
  //

  // (2) create GPU objects
  //
  WGPUBufferDescriptor bufDesc = {};
  // in
  {% for var in MainFunc.FullImpl.InputData %}
  {% if var.IsTexture %}
  //auto {{var.Name}}Img = vk_utils::createImg(device, {{var.Name}}.width(), {{var.Name}}.height(), {{var.Format}}, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
  //size_t {{var.Name}}ImgId = images.size();
  //images.push_back(&{{var.Name}}Img);
  //images2.push_back({{var.Name}}Img.image);
  {% else %}
  bufDesc.size  = {{var.DataSize}}*sizeof({{var.DataType}});
  bufDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
  WGPUBuffer {{var.Name}}GPU = wgpuDeviceCreateBuffer(device, &bufDesc);
  {% endif %}
  {% endfor %}
  // out
  {% for var in MainFunc.FullImpl.OutputData %}
  {% if var.IsTexture %}
  //auto {{var.Name}}Img = vk_utils::createImg(device, {{var.Name}}.width(), {{var.Name}}.height(), {{var.Format}}, outFlags);
  //size_t {{var.Name}}ImgId = images.size();
  //images.push_back(&{{var.Name}}Img);
  //images2.push_back({{var.Name}}Img.image);
  {% else %}
  const size_t {{var.Name}}Size = {{var.DataSize}}*sizeof({{var.DataType}});
  bufDesc.size  = {{var.Name}}Size;
  bufDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
  WGPUBuffer {{var.Name}}GPU = wgpuDeviceCreateBuffer(device, &bufDesc);
  {% endif %}
  {% endfor %}

  SetWulkanInOutFor_{{MainFunc.Name}}({{MainFunc.FullImpl.ArgsOnSetInOut}});

  // (3) copy input data to GPU
  //
  {% for var in MainFunc.FullImpl.InputData %}
  {% if var.IsTexture %}
  //pCopyHelper->UpdateImage({{var.Name}}Img.image, {{var.Name}}.data(), {{var.Name}}.width(), {{var.Name}}.height(), {{var.Name}}.bpp(), VK_IMAGE_LAYOUT_GENERAL);
  {% else %}
  wgpuQueueWriteBuffer(queue, {{var.Name}}GPU, 0, {{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}));
  {% endif %}
  {% endfor %}

  // (4) now execute algorithm on GPU
  //
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
  {{MainFunc.Name}}Cmd(encoder, {{MainFunc.FullImpl.ArgsOnCall}});

  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
  //wgpuCommandEncoderRelease(encoder); //removed function ?
  wgpuQueueSubmit(queue, 1, &cmd);

  // (5) copy output data to CPU
  //
  size_t maxReadSize = 0;
  {% for var in MainFunc.FullImpl.OutputData %}
  {% if var.IsTexture %}
  //pCopyHelper->ReadImage({{var.Name}}Img.image, {{var.Name}}.data(), {{var.Name}}.width(), {{var.Name}}.height(), {{var.Name}}.bpp());
  maxReadSize = std::max(maxReadSize, size_t({{var.Name}}.width()*{{var.Name}}.height()*{{var.Name}}.bpp()));
  {% else %}
  maxReadSize = std::max(maxReadSize, size_t({{var.DataSize}}*sizeof({{var.DataType}})));
  {% endif %}
  {% endfor %}
  
  if(m_readBackBuffer == nullptr || m_readBackSize < maxReadSize)
  {
    WGPUBufferDescriptor readDesc = {};
    readDesc.size    = maxReadSize;
    readDesc.usage   = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    m_readBackBuffer = wgpuDeviceCreateBuffer(device, &readDesc);
  }

  {% for var in MainFunc.FullImpl.OutputData %}
  {% if var.IsTexture %}
  //pCopyHelper->ReadImage({{var.Name}}Img.image, {{var.Name}}.data(), {{var.Name}}.width(), {{var.Name}}.height(), {{var.Name}}.bpp());
  {% else %}
  ReadBufferBack({{var.Name}}GPU, {{var.DataSize}}*sizeof({{var.DataType}}), {{var.Name}});
  {% endif %}
  {% endfor %}

  {% if ContantUBO or UniformUBO %}
  //this->ReadPlainMembers(pCopyHelper);
  {% else %}
  //this->ReadPlainMembers(pCopyHelper); // TODO: implement 'ReadPlainMembers'
  {% endif %}

  // (6) free resources
  //
  
}
{% endif %}
{% endfor %}

{{MainClassName}}{{MainClassSuffix}}::~{{MainClassName}}{{MainClassSuffix}}()
{
  
}