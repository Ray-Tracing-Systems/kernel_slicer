//define sms tu include CUDA-enabled verion of LiteMath
#include "LiteMath.h"
#include <thrust/device_vector.h>

// if CUDA + Slang, include kernels which are generated from slang
{% for Kernel in Kernels %}
//#include "{{MainClassName}}{{MainClassSuffix}}_{{Kernel.Name}}_generated.cuh"
{% endfor %}

// if Pure CUDA, put kernels directly here
{% for Kernel in Kernels %}
//define {{Kernel.Name}} here ... 
{% endfor %}

using thrust::device_vector;

class {{MainClassName}}{{MainClassSuffix}} : public {{MainClassName}}
{
public:

  {% for ctorDecl in Constructors %}
  {% if ctorDecl.NumParams == 0 %}
  {{ctorDecl.ClassName}}{{MainClassSuffix}}()
  {
    {% if HasPrefixData %}
    if({{PrefixDataName}} == nullptr)
      {{PrefixDataName}} = std::make_shared<{{PrefixDataClass}}>();
    {% endif %}
  }
  {% else %}
  {{ctorDecl.ClassName}}{{MainClassSuffix}}({{ctorDecl.Params}}) : {{ctorDecl.ClassName}}({{ctorDecl.PrevCall}})
  {
    {% if HasPrefixData %}
    if({{PrefixDataName}} == nullptr)
      {{PrefixDataName}} = std::make_shared<{{PrefixDataClass}}>();
    {% endif %}
  }
  {% endif %}
  {% endfor %}
  
  void InitCudaObjects(size_t a_maxThreadsGenerated);
  
protected:
  {% for Vector in VectorMembers %}
  device_vector<{{Vector.DataType}}> {{Vector.Name}}_dev;
  {% endfor %}
};

{% for ctorDecl in Constructors %}
{% if ctorDecl.NumParams == 0 %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}(size_t a_maxThreadsGenerated)
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>();
  pObj->InitCudaObjects(a_maxThreadsGenerated);
  return pObj;
}
{% else %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}({{ctorDecl.Params}})
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>({{ctorDecl.PrevCall}}, size_t a_maxThreadsGenerated);
  pObj->InitCudaObjects(a_maxThreadsGenerated);
  return pObj;
}
{% endif %}
{% endfor %}

void {{MainClassName}}{{MainClassSuffix}}::InitCudaObjects(size_t a_maxThreadsGenerated)
{
  
}
