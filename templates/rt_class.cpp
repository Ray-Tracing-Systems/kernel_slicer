#include <vector>
#include <memory>
#include <limits>

#include <cassert>

#include "vulkan_basics.h"
#include "{{IncludeClassDecl}}"

{{MainClassName}}_Generated::~{{MainClassName}}_Generated()
{
  m_pMaker = nullptr;

## for Kernel in Kernels
  vkDestroyDescriptorSetLayout(device, {{Kernel.Name}}DSLayout, nullptr);
  {{Kernel.Name}}DSLayout = VK_NULL_HANDLE;
## endfor
  vkDestroyDescriptorPool(device, m_dsPool, NULL); m_dsPool = VK_NULL_HANDLE;

## for MainFunc in MainFunctions
## for Buffer in MainFunc.LocalVarsBuffers
  vkDestroyBuffer(device, {{Buffer.Name}}Buffer, nullptr);
## endfor

## endfor
  vkDestroyBuffer(device, m_classDataBuffer, nullptr);

## for Var in ClassVectorVars
  vkDestroyBuffer(device, m_vdata.{{Var.Name}}Buffer, nullptr);
## endfor

  if(m_allMem != VK_NULL_HANDLE)
    vkFreeMemory(device, m_allMem, nullptr);
  
  if(m_vdata.m_vecMem != VK_NULL_HANDLE)
    vkFreeMemory(device, m_vdata.m_vecMem, nullptr);
}

void {{MainClassName}}_Generated::InitHelpers()
{
  vkGetPhysicalDeviceProperties(physicalDevice, &m_devProps);
  m_pMaker = std::make_unique<vkfw::ComputePipelineMaker>();
}

## for Kernel in Kernels
VkDescriptorSetLayout {{MainClassName}}_Generated::Create{{Kernel.Name}}DSLayout()
{
  VkDescriptorSetLayoutBinding dsBindings[{{Kernel.ArgCount}}+1] = {};
  
## for KernelARG in Kernel.Args
  // binding for {{KernelARG.Name}}
  dsBindings[{{KernelARG.Id}}].binding            = {{KernelARG.Id}};
  dsBindings[{{KernelARG.Id}}].descriptorType     = {{KernelARG.Type}};
  dsBindings[{{KernelARG.Id}}].descriptorCount    = 1;
  dsBindings[{{KernelARG.Id}}].stageFlags         = {{KernelARG.Flags}};
  dsBindings[{{KernelARG.Id}}].pImmutableSamplers = nullptr;

## endfor
  // binding for POD members stored in m_classDataBuffer
  dsBindings[{{Kernel.ArgCount}}].binding            = {{Kernel.ArgCount}};
  dsBindings[{{Kernel.ArgCount}}].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  dsBindings[{{Kernel.ArgCount}}].descriptorCount    = 1;
  dsBindings[{{Kernel.ArgCount}}].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  dsBindings[{{Kernel.ArgCount}}].pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
  descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.bindingCount = uint32_t({{Kernel.ArgCount}}+1);
  descriptorSetLayoutCreateInfo.pBindings    = dsBindings;
  
  VkDescriptorSetLayout layout = nullptr;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &layout));
  return layout;
}

## endfor

void {{MainClassName}}_Generated::InitKernels(const char* a_filePath, uint32_t a_blockSizeX, uint32_t a_blockSizeY, uint32_t a_blockSizeZ)
{
  VkSpecializationMapEntry specializationEntries[3] = {};
  {
    specializationEntries[0].constantID = 0;
    specializationEntries[0].offset     = 0;
    specializationEntries[0].size       = sizeof(uint32_t);
  
    specializationEntries[1].constantID = 1;
    specializationEntries[1].offset     = sizeof(uint32_t);
    specializationEntries[1].size       = sizeof(uint32_t);
  
    specializationEntries[2].constantID = 2;
    specializationEntries[2].offset     = 2 * sizeof(uint32_t);
    specializationEntries[2].size       = sizeof(uint32_t);
  }

  uint32_t specializationData[3] = {a_blockSizeX, a_blockSizeY, a_blockSizeZ};
  VkSpecializationInfo specsForWGSize = {};
  {
    specsForWGSize.mapEntryCount = 3;
    specsForWGSize.pMapEntries   = specializationEntries;
    specsForWGSize.dataSize      = 3 * sizeof(uint32_t);
    specsForWGSize.pData         = specializationData;
  }
 
## for Kernel in Kernels
  {{Kernel.Name}}DSLayout = Create{{Kernel.Name}}DSLayout();
  m_pMaker->CreateShader(device, a_filePath, &specsForWGSize, "{{Kernel.OriginalName}}");

  {{Kernel.Name}}Layout   = m_pMaker->MakeLayout(device, {{Kernel.Name}}DSLayout, sizeof(uint32_t)*4);
  {{Kernel.Name}}Pipeline = m_pMaker->MakePipeline(device);   

## endfor
}


void {{MainClassName}}_Generated::UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
{
## for Var in ClassVars
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, {{Var.Offset}}, &{{Var.Name}}, {{Var.Size}});
## endfor
}

void {{MainClassName}}_Generated::UpdateVectorMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
{
  const size_t maxAllowedSize = std::numeric_limits<uint32_t>::max();

## for Var in ClassVectorVars
  {
    uint32_t sizeOfVector     = uint32_t( {{Var.Name}}.size() ); assert( {{Var.Name}}.size() < maxAllowedSize );
    uint32_t capacityOfVector = uint32_t( {{Var.Name}}.capacity() ); assert( {{Var.Name}}.capacity() < maxAllowedSize );
    a_pCopyEngine->UpdateBuffer(m_classDataBuffer, {{Var.SizeOffset}}, &sizeOfVector, sizeof(uint32_t));
    a_pCopyEngine->UpdateBuffer(m_classDataBuffer, {{Var.CapacityOffset}}, &capacityOfVector, sizeof(uint32_t));
  }

## endfor

## for Var in ClassVectorVars
  a_pCopyEngine->UpdateBuffer(m_vdata.{{Var.Name}}Buffer, 0, {{Var.Name}}.data(), {{Var.Name}}.size()*sizeof({{Var.TypeOfData}}) );
## endfor
}

void {{MainClassName}}_Generated::InitBuffers(size_t a_maxThreadsCount)
{
  std::vector<VkBuffer> allBuffers;

## for MainFunc in MainFunctions  
## for Buffer in MainFunc.LocalVarsBuffers
  {{Buffer.Name}}Buffer = vkfw::CreateBuffer(device, sizeof({{Buffer.Type}})*a_maxThreadsCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  allBuffers.push_back({{Buffer.Name}}Buffer);
## endfor
## endfor

  m_classDataBuffer = vkfw::CreateBuffer(device, {{AllClassVarsSize}},  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  allBuffers.push_back(m_classDataBuffer);
  
  if(allBuffers.size() > 0)
    m_allMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, allBuffers);
}

void {{MainClassName}}_Generated::InitMemberBuffers()
{
  std::vector<VkBuffer> memberVectors;
## for Var in ClassVectorVars
  m_vdata.{{Var.Name}}Buffer = vkfw::CreateBuffer(device, {{Var.Name}}.capacity()*sizeof({{Var.TypeOfData}}), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  memberVectors.push_back(m_vdata.{{Var.Name}}Buffer);
## endfor
  
  if(memberVectors.size() > 0)
    m_vdata.m_vecMem = vkfw::AllocateAndBindWithPadding(device, physicalDevice, memberVectors);
}

## for Kernel in Kernels
void {{MainClassName}}_Generated::{{Kernel.Decl}}
{
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}Pipeline);
  
  uint32_t pcData[4] = { {{Kernel.tidX}}, {{Kernel.tidY}}, {{Kernel.tidZ}}, 1};
  vkCmdPushConstants(m_currCmdBuffer, TestColorLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)*4, pcData);
  
  vkCmdDispatch(m_currCmdBuffer, {{Kernel.tidX}}/m_blockSize[0], {{Kernel.tidY}}/m_blockSize[1], {{Kernel.tidZ}}/m_blockSize[2]);

  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
  vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
}

## endfor

## for MainFunc in MainFunctions
{{MainFunc.MainFuncCmd}}

## endfor

