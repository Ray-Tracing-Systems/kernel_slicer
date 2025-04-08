#pragma once

#include <vector>
#include <memory>
#include <string>
#include <array>

#include "vk_pipeline.h"
#include "vk_buffers.h"
#include "vk_utils.h"
#include "vk_copy.h"
#include "vk_context.h"

{% if length(TextureMembers) > 0 or length(ClassTexArrayVars) > 0 %}
#include "Image2d.h"
using LiteImage::Image2D;
using LiteImage::Sampler;
using namespace LiteMath;
{% endif %}
#include "{{MainInclude}}"
{% for Include in AdditionalIncludes %}
#include "{{Include}}"
{% endfor %}
## for Decl in ClassDecls  
{% if Decl.InClass and Decl.IsType %}
using {{Decl.Type}} = {{MainClassName}}::{{Decl.Type}}; // for passing this data type to UBO
{% endif %}
## endfor
{% for SetterDecl in SettersDecl %}  
{{SetterDecl}}

{% endfor %}

// how to use generated class '{{MainClassName}}{{MainClassSuffix}}' and GPU API 'I{{MainClassName}}{{MainClassSuffix}}':
// (0) auto pImplGPU = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>(); 
//                                                                                  // or use Create{{MainClassName}}{{MainClassSuffix}}(...) function
// then you have 2 basic variants of working with GPU API:

// (1) simple one:
// (1.1) pImplGPU->SetVulkanContext(ctx);                                           // or use Create{{MainClassName}}{{MainClassSuffix}}(...) function
// (1.2) pImplGPU->InitVulkanObjects(device, physicalDevice, MAX_THREADS);          // or use Create{{MainClassName}}{{MainClassSuffix}}(...) function
// (1.2) do so some stuff on CPU in base class
// (1.3) pImplGPU->CommitDeviceData();            // use implicit commit
// (1.4) now you can work BOTH with CPU and GPU API

// (2) more explicit:
// (2.1) pImplGPU->InitVulkanObjects(device, physicalDevice, MAX_THREADS);
// (2.2) do so some stuff on CPU in base class
// (2.3) pImplGPU->CommitDeviceData(pCopyHelper); // use explicit commit
// (2.4) now you can work with GPU API only

// (3) finally you may use std::dynamic_pointer_cast to get GPU API of 'I{{MainClassName}}{{MainClassSuffix}}' class:
// pGPUAPI = std::dynamic_pointer_cast<I{{MainClassName}}{{MainClassSuffix}}>(pImplGPU);

class I{{MainClassName}}{{MainClassSuffix}} 
{
public:
  I{{MainClassName}}{{MainClassSuffix}}(){}
  virtual ~I{{MainClassName}}{{MainClassSuffix}}(){}
  {% if HasNameFunc %}
  virtual const char* Name() const { return "I{{MainClassName}}{{MainClassSuffix}}";}
  {% endif %}
  
  virtual void SetVulkanContext(vk_utils::VulkanContext a_ctx) = 0;
  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount) = 0;  
  virtual void CommitDeviceData(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine) // explicit commit
  {
    {% if HasPrefixData %}
    UpdatePrefixPointers(); 
    {% endif %}
    InitDeviceData();
    UpdateDeviceData(a_pCopyEngine);
  }  

## for MainFunc in MainFunctions
  virtual void SetVulkanInOutFor_{{MainFunc.Name}}(
## for Arg in MainFunc.InOutVars
    {% if Arg.IsTexture %}
    VkImage     {{Arg.Name}}Text,
    VkImageView {{Arg.Name}}View,
    {% else %}
    VkBuffer {{Arg.Name}}Buffer,
    size_t   {{Arg.Name}}Offset,
    {% endif %}
## endfor
    uint32_t dummyArgument = 0) = 0;

  virtual void {{MainFunc.Name}}Cmd(VkCommandBuffer a_commandBuffer, {% for Arg in MainFunc.InOutVarsPod %}{{Arg.Type}} {{Arg.Name}}{% if loop.index1 != MainFunc.InOutVarsNumPod %}, {% endif %}{% endfor %}) { {{MainFunc.Name}}Cmd(a_commandBuffer, {% for Arg in MainFunc.InOutVarsAll %}{% if Arg.IsPointer or Arg.IsTexture %}nullptr{% else %}{{Arg.Name}}{% endif %}{% if loop.index1 != MainFunc.InOutVarsNumAll %}, {% endif %}{% endfor %}); }

## endfor
  {% for SetterFunc in SetterFuncs %}  
  {{SetterFunc}}
  {% endfor %}
  virtual void InitDeviceData() = 0;
  virtual void UpdateDeviceData(std::shared_ptr<vk_utils::ICopyEngine> a_pCopyEngine) = 0;
protected:
  {% for MainFunc in MainFunctions %}
  virtual {{MainFunc.ReturnType}} {{MainFunc.Decl}} = 0;
  {% endfor %}
};
