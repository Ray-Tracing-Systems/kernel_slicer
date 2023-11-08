#ifndef MAIN_CLASS_DECL_{{MainClassName}}_H
#define MAIN_CLASS_DECL_{{MainClassName}}_H

#include <vector>
#include <memory>
#include <string>

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
#include "include/{{UBOIncl}}"
{% for SetterDecl in SettersDecl %}  
{{SetterDecl}}

{% endfor %}
class I{{MainClassName}}{{MainClassSuffix}} : public {{MainClassName}}
{
public:
  
  virtual ~{{MainClassName}}{{MainClassSuffix}}(){}
  {% if HasNameFunc %}
  virtual const char* Name() const { return "I{{MainClassName}}{{MainClassSuffix}}";}
  {% endif %}
  virtual void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount) = 0;

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
    
  virtual {{MainFunc.ReturnType}} {{MainFunc.Decl}};
## endfor

  {% for SetterFunc in SetterFuncs %}  
  {{SetterFunc}}
  {% endfor %}
  {% if HasCommitDeviceFunc %}
  virtual void CommitDeviceData() {} // kslicer override this virtual function in derived class
  {% endif %}
  {% if HasGetTimeFunc %}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // kslicer override this virtual function in derived class
  {% endif %}
};

#endif
