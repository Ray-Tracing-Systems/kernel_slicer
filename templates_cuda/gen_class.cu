
#include "LiteMath.h"
#include <extended/lm_device_vector.h> // also from LiteMath

namespace {{MainClassName}}{{MainClassSuffix}}_DEV
{
  {% for Vector in VectorMembers %}
  __device__ LiteMathExtended::device_vector<{{Vector.DataType}}> {{Vector.Name}};
  {% endfor %}
  {% for Field in UBO.UBOStructFields %}
  {% if Field.IsDummy %} 
  __device__ uint {{Field.Name}}; 
  {% else %}
  {% if not Field.IsContainerInfo %}
  __device__ {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %};
  {% endif %}
  {% endif %}
  {% endfor %}
  // if Pure CUDA, put kernels directly here
  {% for Kernel in Kernels %}
  //define {{Kernel.Name}} here ... 
  {% endfor %}
};

#include <memory>
#include <cstdint>
#include <cassert>
#include <chrono>
#include "{{MainInclude}}"
{% for Include in AdditionalIncludes %}
#include "{{Include}}"
{% endfor %}

//#include <thrust/device_vector.h> // if use real thrust
//using thrust::device_vector;      // if use real thrust
using LiteMathExtended::device_vector;

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
  
  void CommitDeviceData() override;

  void CopyUBOToDevice(bool a_updateVectorSize = true);
  void CopyUBOFromDevice();
  void UpdateDeviceVectors();

  {% for Kernel in Kernels %}
  void {{Kernel.OriginalDecl}} override;
  {% endfor %}
  
  {% for MainFunc in MainFunctions %}
  {{MainFunc.ReturnType}} {{MainFunc.MainFuncDeclCmd}} override;
  {% endfor %}

protected:
  {% for Vector in VectorMembers %}
  device_vector<{{Vector.DataType}}> {{Vector.Name}}_dev;
  {% endfor %}
};

{% for ctorDecl in Constructors %}
{% if ctorDecl.NumParams == 0 %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}()
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>();
  return pObj;
}
{% else %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}({{ctorDecl.Params}})
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}>({{ctorDecl.PrevCall}});
  return pObj;
}
{% endif %}
{% endfor %}

void {{MainClassName}}{{MainClassSuffix}}::CopyUBOToDevice(bool a_updateVectorSize)
{
  {% for Var in ClassVars %}
  {% if Var.IsArray %}
  {% if Var.HasPrefix %}
  cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, pUnderlyingImpl->{{Var.CleanName}}, sizeof(pUnderlyingImpl->{{Var.CleanName}}));
  {% else %}
  cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, {{Var.Name}}, sizeof({{Var.Name}}));
  {% endif %}
  {% else %}
  {% if Var.HasPrefix %}
  m_uboData.{{Var.Name}} = pUnderlyingImpl->{{Var.CleanName}};
  cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, &pUnderlyingImpl->{{Var.CleanName}}, sizeof(pUnderlyingImpl->{{Var.CleanName}}));
  {% else %}
  cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, &{{Var.Name}}, sizeof({{Var.Name}}));
  {% endif %}
  {% endif %}
  {% endfor %}
  if(a_updateVectorSize)
  {
    using size_type = LiteMathExtended::device_vector<int>::size_type;
    {% for Var in ClassVectorVars %}
    {
      const size_type currSize = {{Var.Name}}_dev.size();
      cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}.m_size, &currSize, sizeof(size_type));
    }
    {% endfor %}
  }
}

void {{MainClassName}}{{MainClassSuffix}}::CopyUBOFromDevice()
{
  //cudaMemcpyFromSymbol(&h_globalVar, globalVar, sizeof(int));
  {% for Var in ClassVars %}
  {% if Var.IsArray %}
  {% if Var.HasPrefix %}
  cudaMemcpyFromSymbol(pUnderlyingImpl->{{Var.CleanName}}, {{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, sizeof(pUnderlyingImpl->{{Var.CleanName}}));
  {% else %}
  cudaMemcpyFromSymbol({{Var.Name}}, {{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, sizeof({{Var.Name}}));
  {% endif %}
  {% else %}
  {% if Var.HasPrefix %}
  m_uboData.{{Var.Name}} = pUnderlyingImpl->{{Var.CleanName}};
  cudaMemcpyFromSymbol(&pUnderlyingImpl->{{Var.CleanName}}, {{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, sizeof(pUnderlyingImpl->{{Var.CleanName}}));
  {% else %}
  cudaMemcpyFromSymbol(&{{Var.Name}}, {{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, sizeof({{Var.Name}}));
  {% endif %}
  {% endif %}
  {% endfor %}
  using size_type = LiteMathExtended::device_vector<int>::size_type;
  {% for Var in ClassVectorVars %}
  {
    size_type currSize = 0;
    cudaMemcpyFromSymbol(&currSize, {{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}.m_size, sizeof(size_type));
    {{Var.Name}}.resize(currSize);
  }
  {% endfor %}
}

void {{MainClassName}}{{MainClassSuffix}}::UpdateDeviceVectors()
{
  using size_type = LiteMathExtended::device_vector<int>::size_type;
  {% for Var in VectorMembers %}
  {
    const size_type currSize = {{Var.Name}}_dev.size();
    const size_type currCapa = {{Var.Name}}_dev.capacity();
    const void*     currPtr  = {{Var.Name}}_dev.data();
    cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}.m_data,     &currPtr,  sizeof(void*));
    cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}.m_size    , &currSize, sizeof(size_type));
    cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}.m_capacity, &currCapa, sizeof(size_type));
  }
  {% endfor %}
}

void {{MainClassName}}{{MainClassSuffix}}::CommitDeviceData()
{ 
  {% for Var in VectorMembers %}
  {{Var.Name}}_dev.assign({{Var.Name}}.begin(), {{Var.Name}}.end());
  {% endfor %}
  UpdateDeviceVectors();
  CopyUBOToDevice(false);
}

{% for Kernel in Kernels %}
void {{MainClassName}}{{MainClassSuffix}}::{{Kernel.OriginalDecl}}
{
  // call actual kernel here
}

{% endfor %}
{% for MainFunc in MainFunctions %}
{{MainFunc.ReturnType}} {{MainClassName}}{{MainClassSuffix}}::{{MainFunc.MainFuncDeclCmd}}
{
  {% for var in MainFunc.FullImpl.InputData %}
  {{var.DataType}}* {{var.Name}}Host = {{var.Name}};
  {% endfor %}
  {% for var in MainFunc.FullImpl.OutputData %}
  {{var.DataType}}* {{var.Name}}Host = {{var.Name}};
  {% endfor %}

  {% for var in MainFunc.FullImpl.InputData %}
  cudaMalloc(&{{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}));
  {% endfor %}
  {% for var in MainFunc.FullImpl.OutputData %}
  cudaMalloc(&{{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}));
  {% endfor %}
  {% for var in MainFunc.FullImpl.InputData %}
  cudaMemcpy((void*){{var.Name}}, {{var.Name}}Host, {{var.DataSize}}*sizeof({{var.DataType}}), cudaMemcpyHostToDevice);
  {% endfor %}

  CopyUBOToDevice(true);
  {{MainFunc.MainFuncTextCmd}}
  CopyUBOFromDevice();
  {% for var in MainFunc.FullImpl.OutputData %}
  cudaMemcpy({{var.Name}}Host, {{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}), cudaMemcpyDeviceToHost);
  {% endfor %}
}

{% endfor %}