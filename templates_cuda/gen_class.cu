
#include "LiteMath.h"
#include <extended/lm_device_vector.h> // also from LiteMath
#include "{{MainInclude}}"
#include <cfloat>

namespace {{MainClassName}}{{MainClassSuffix}}_DEV
{
  using _Bool = bool;
  {% for Decl in ClassDecls %}
  {% if Decl.InClass and Decl.IsType %}
  using {{Decl.Type}} = {{MainClassName}}::{{Decl.Type}}; // for passing this data type to kernels
  {% endif %}
  {% endfor %}

  {% for LocalFunc in LocalFunctions %} 
  __device__ {{LocalFunc}}

  {% endfor %}
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
  
  {% for MembFunc in AllMemberFunctions %}
  __device__ {{MembFunc.Decl}};
  {% endfor %}
  {% for MembFunc in AllMemberFunctions %}

  __device__ {{MembFunc.Text}}
  {% endfor %}
  {% if UseSubGroups %}
  template<typename T>
  __device__ void WarpReduceSum(volatile T* sdata, int tid) 
  {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
  }

  template<typename T>
  __device__ void WarpReduceMin(volatile T* sdata, int tid) 
  {
    sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    sdata[tid] = min(sdata[tid], sdata[tid + 1]);
  }

  template<typename T>
  __device__ void WarpReduceMax(volatile T* sdata, int tid) 
  {
    sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    sdata[tid] = max(sdata[tid], sdata[tid + 1]);
  }
  
  {% endif %}
  __device__ float atomicMin(float* address, float val) 
  {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float current_val = __int_as_float(old);
        if (val >= current_val) 
          break;  // Если новое значение не меньше, выходим
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (expected != old);
    return __int_as_float(old);
  }

  __device__ float atomicMax(float* address, float val) 
  {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float current_val = __int_as_float(old);
        if (val <= current_val) 
          break;  // Если новое значение не больше, выходим
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (expected != old);
    return __int_as_float(old);
  }
 
  {% for Kernel in KernelList %}
  __global__ void {{Kernel.Name}}({%for Arg in Kernel.OriginalArgs %}{{Arg.Type}} {{Arg.Name}}{% if loop.index != Kernel.LastArgAll %}, {% endif %}{% endfor %})
  {
    {% if not Kernel.IsSingleThreaded %}
    const uint _threadID[3] = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z
    };
    {% for TID in Kernel.ThreadIds %}
    {% if TID.Simple %}
    const {{TID.Type}} {{TID.Name}} = {{TID.Type}}(_threadID[{{ loop.index }}]); 
    {% else %}
    const {{TID.Type}} {{TID.Name}} = {{TID.Start}} + {{TID.Type}}(_threadID[{{ loop.index }}])*{{TID.Stride}}; 
    {% endif %}
    {% endfor %}
    bool runThisThread = true;
    {% if not Kernel.EnableBlockExpansion %}
    {% if Kernel.IsIndirect %}
    {% if Kernel.threadDim == 3 %}
    if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}} + {{Kernel.CondLE2}} || {{Kernel.threadName3}} >= {{Kernel.IndirectSizeZ}} + {{Kernel.CondLE3}})
      runThisThread = false;
    {% else if Kernel.threadDim == 2 %}
    if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}} + {{Kernel.CondLE2}})
      runThisThread = false;
    {% else %}
    if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}})
      runThisThread = false;
    {% endif %}
    {% else %}
    {% if Kernel.threadDim == 3 %}
    if({{Kernel.threadName1}} >= {{Kernel.threadSZName1}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= {{Kernel.threadSZName2}} + {{Kernel.CondLE2}} || {{Kernel.threadName3}} >= {{Kernel.threadSZName3}} + {{Kernel.CondLE3}})
      runThisThread = false;
    {% else if Kernel.threadDim == 2 %}
    if({{Kernel.threadName1}} >= {{Kernel.threadSZName1}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= {{Kernel.threadSZName2}} + {{Kernel.CondLE2}})
      runThisThread = false;
    {% else %}
    if({{Kernel.threadName1}} >= {{Kernel.threadSZName1}} + {{Kernel.CondLE1}})
      runThisThread = false;
    {% endif %}
    {% endif %} {# /* if Kernel.IsIndirect  */ #}
    {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                        
    {% include "inc_red_init.cu" %}
    {% endif %} 
    {% endif %} {# /* if not Kernel.EnableBlockExpansion */ #}
    if(runThisThread) 
    {
    {% endif %} {# /* if not Kernel.IsSingleThreaded */ #}
    {{Kernel.Source}}
    {% if not Kernel.EnableBlockExpansion and not Kernel.IsSingleThreaded %}
    }
    {% endif %}
    {% if Kernel.HasEpilog %}
    // GENERATED EPILOG:
    //
    {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                      
    {% include "inc_red_finish.cu" %}
    {% endif %}
    {% endif %} {# /* END of 'if Kernel.HasEpilog'  */ #}
  }

  {% if Kernel.IsIndirect and not Kernel.IsSingleThreaded %}
  __global__ void {{Kernel.Name}}_Indirect({%for Arg in Kernel.OriginalArgs %}{{Arg.Type}} {{Arg.Name}}{% if loop.index != Kernel.LastArgAll %}, {% endif %}{% endfor %})
  {
    dim3 blocksNum, blockSize;
    blocksNum.x = ({{Kernel.IndirectSizeX}} - {{Kernel.IndirectStartX}} + {{Kernel.WGSizeX}} - 1)/{{Kernel.WGSizeX}};
    {% if Kernel.threadDim == 2 %}
    blocksNum.y = ({{Kernel.IndirectSizeY}} - {{Kernel.IndirectStartY}} + {{Kernel.WGSizeY}} - 1)/{{Kernel.WGSizeY}};
    {% endif %}
    {% if Kernel.threadDim == 3 %}
    blocksNum.z = ({{Kernel.IndirectSizeZ}} - {{Kernel.IndirectStartZ}} + {{Kernel.WGSizeZ}} - 1)/{{Kernel.WGSizeZ}};
    {% endif %}
    blockSize.x = {{Kernel.WGSizeX}};
    blockSize.y = {{Kernel.WGSizeY}};
    blockSize.z = {{Kernel.WGSizeZ}};
    {{Kernel.Name}}<<<blocksNum, blockSize>>>({%for Arg in Kernel.OriginalArgs %}{{Arg.Name}}{% if loop.index != Kernel.LastArgAll %}, {% endif %}{% endfor %});
  }
  
  {% endif %}
  {% endfor %}
};

#include <memory>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <vector>
#include <string>
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
  
  virtual ~{{MainClassName}}{{MainClassSuffix}}()
  {
    {% for Vector in VectorMembers %}
    {{Vector.Name}}_dev.resize(0);
    {{Vector.Name}}_dev.shrink_to_fit(); 
    {% endfor %}
  }

  void CommitDeviceData() override;
  {% if HasGetTimeFunc %}
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override;
  {% endif %}
  void CopyUBOToDevice(bool a_updateVectorSize = true);
  void CopyUBOFromDevice(bool a_updateVectorSize = true);
  void UpdateDeviceVectors();

  {% for Kernel in Kernels %}
  void {{Kernel.OriginalDecl}} override;
  {% endfor %}
  
  {% for MainFunc in MainFunctions %}
  {{MainFunc.ReturnType}} {{MainFunc.Name}}({%for Arg in MainFunc.InOutVarsAll %}{%if Arg.IsConst %}const {%endif%}{{Arg.Type}} {{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %}) override;
  virtual {{MainFunc.ReturnType}} {{MainFunc.Name}}GPU({%for Arg in MainFunc.InOutVarsAll %}{%if Arg.IsConst %}const {%endif%}{{Arg.Type}} {{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %});
  {% endfor %}

protected:
  {% for Vector in VectorMembers %}
  device_vector<{{Vector.DataType}}> {{Vector.Name}}_dev;
  {% endfor %}
  {% for MainFunc in MainFunctions %}
  float m_exTime{{MainFunc.Name}}[4] = {0,0,0,0};
  {% endfor %}
};

class {{MainClassName}}{{MainClassSuffix}}DEV : public {{MainClassName}}{{MainClassSuffix}}
{
public:

  {% for ctorDecl in Constructors %}
  {% if ctorDecl.NumParams == 0 %}
  {{ctorDecl.ClassName}}{{MainClassSuffix}}DEV() {}
  {% else %}
  {{ctorDecl.ClassName}}{{MainClassSuffix}}DEV({{ctorDecl.Params}}) : {{ctorDecl.ClassName}}({{ctorDecl.PrevCall}}) {}
  {% endif %}
  {% endfor %}
  {% for MainFunc in MainFunctions %}
  
  {{MainFunc.ReturnType}} {{MainFunc.Name}}({%for Arg in MainFunc.InOutVarsAll %}{%if Arg.IsConst %}const {%endif%}{{Arg.Type}} {{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %}) override {
    {% if MainFunc.IsVoid %}
    {{MainFunc.Name}}GPU({%for Arg in MainFunc.InOutVarsAll %}{{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %});
    {% else %}
    return {{MainFunc.Name}}GPU({%for Arg in MainFunc.InOutVarsAll %}{{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %});
    {% endif %}
  }
  {% endfor %}

protected:
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
{% for ctorDecl in Constructors %}
{% if ctorDecl.NumParams == 0 %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}_DEV()
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}DEV>();
  return pObj;
}
{% else %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}{{MainClassSuffix}}_DEV({{ctorDecl.Params}})
{
  auto pObj = std::make_shared<{{MainClassName}}{{MainClassSuffix}}DEV>({{ctorDecl.PrevCall}});
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
    {% for Var in VectorMembers %}
    cudaMemcpyToSymbol({{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, &{{Var.Name}}_dev, sizeof(LiteMathExtended::device_vector<{{Var.DataType}}>));
    {% endfor %}
  }
}

void {{MainClassName}}{{MainClassSuffix}}::CopyUBOFromDevice(bool a_updateVectorSize)
{
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
  if(a_updateVectorSize)
  {
    {% for Var in VectorMembers %}
    cudaMemcpyFromSymbol(&{{Var.Name}}_dev, {{MainClassName}}{{MainClassSuffix}}_DEV::{{Var.Name}}, sizeof(LiteMathExtended::device_vector<{{Var.DataType}}>));
    if({{Var.Name}}.size() != {{Var.Name}}_dev.size())
      {{Var.Name}}.resize({{Var.Name}}_dev.size());
    {% endfor %}
  }
}

void {{MainClassName}}{{MainClassSuffix}}::UpdateDeviceVectors() 
{
  {% for Var in VectorMembers %}
  {{Var.Name}}_dev.reserve({{Var.Name}}.capacity());
  {{Var.Name}}_dev.assign({{Var.Name}}.begin(), {{Var.Name}}.end());
  {% endfor %}
}

void {{MainClassName}}{{MainClassSuffix}}::CommitDeviceData()
{
  UpdateDeviceVectors();
  CopyUBOToDevice(true);
}

{% for Kernel in Kernels %}
void {{MainClassName}}{{MainClassSuffix}}::{{Kernel.OriginalDecl}}
{
  {% if Kernel.HasLoopInit %}
  {{MainClassName}}{{MainClassSuffix}}_DEV::{{Kernel.OriginalName}}_Init<<<1,1>>>({%for Arg in Kernel.OriginalArgs %}{{Arg.Name}}{% if loop.index != Kernel.LastArgAll %}, {% endif %}{% endfor %});
  {% endif %}
  {% if Kernel.IsIndirect %}
  {{MainClassName}}{{MainClassSuffix}}_DEV::{{Kernel.OriginalName}}_Indirect<<<1, 1>>>({%for Arg in Kernel.OriginalArgs %}{{Arg.Name}}{% if loop.index != Kernel.LastArgAll %}, {% endif %}{% endfor %});
  cudaDeviceSynchronize(); // do we need to wait here? 
  {% else %}
  dim3 block({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}});
  dim3 grid(({{Kernel.tidX}} + block.x - 1) / block.x, ({{Kernel.tidY}} + block.y - 1) / block.y, ({{Kernel.tidZ}} + block.z - 1) / block.z);
  {{MainClassName}}{{MainClassSuffix}}_DEV::{{Kernel.OriginalName}}<<<grid, block>>>({%for Arg in Kernel.OriginalArgs %}{{Arg.Name}}{% if loop.index != Kernel.LastArgAll %}, {% endif %}{% endfor %});
  {% endif %}
  {% if Kernel.HasLoopFinish %}
  {{MainClassName}}{{MainClassSuffix}}_DEV::{{Kernel.OriginalName}}_Finish<<<1,1>>>({%for Arg in Kernel.OriginalArgs %}{{Arg.Name}}{% if loop.index != Kernel.LastArgAll %}, {% endif %}{% endfor %});
  {% endif %}
}

{% endfor %}
{% for MainFunc in MainFunctions %}
{{MainFunc.ReturnType}} {{MainClassName}}{{MainClassSuffix}}::{{MainFunc.Name}}GPU({%for Arg in MainFunc.InOutVarsAll %}{%if Arg.IsConst %}const {%endif%}{{Arg.Type}} {{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %})
{
  {{MainFunc.MainFuncTextCmd}}
}

{{MainFunc.ReturnType}} {{MainClassName}}{{MainClassSuffix}}::{{MainFunc.Name}}({%for Arg in MainFunc.InOutVarsAll %}{%if Arg.IsConst %}const {%endif%}{{Arg.Type}} {{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %})
{
  {% for var in MainFunc.FullImpl.InputData %}
  {{var.DataType}}* {{var.Name}}Host = {{var.Name}};
  {% endfor %}
  {% for var in MainFunc.FullImpl.OutputData %}
  {{var.DataType}}* {{var.Name}}Host = {{var.Name}};
  {% endfor %}
  
  cudaEvent_t _start, _stop;
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);
  
  cudaEventRecord(_start);
  {% for var in MainFunc.FullImpl.InputData %}
  cudaMalloc(&{{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}));
  {% endfor %}
  {% for var in MainFunc.FullImpl.OutputData %}
  cudaMalloc(&{{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}));
  {% endfor %}
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&m_exTime{{MainFunc.Name}}[3], _start, _stop);
  
  cudaEventRecord(_start);
  {% for var in MainFunc.FullImpl.InputData %}
  cudaMemcpy((void*){{var.Name}}, {{var.Name}}Host, {{var.DataSize}}*sizeof({{var.DataType}}), cudaMemcpyHostToDevice);
  {% endfor %}
  CopyUBOToDevice(true);
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&m_exTime{{MainFunc.Name}}[1], _start, _stop);
  
  cudaEventRecord(_start);
  {% if MainFunc.IsVoid %}
  {{MainFunc.Name}}GPU({%for Arg in MainFunc.InOutVarsAll %}{{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %});
  {% else %}
  auto _resFromGPU = {{MainFunc.Name}}GPU({%for Arg in MainFunc.InOutVarsAll %}{{Arg.Name}}{% if loop.index != MainFunc.InOutVarsLast %}, {% endif %}{% endfor %});
  {% endif %}
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&m_exTime{{MainFunc.Name}}[0], _start, _stop);
  
  cudaEventRecord(_start);
  CopyUBOFromDevice();
  {% for var in MainFunc.FullImpl.OutputData %}
  cudaMemcpy({{var.Name}}Host, {{var.Name}}, {{var.DataSize}}*sizeof({{var.DataType}}), cudaMemcpyDeviceToHost);
  {% endfor %}
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&m_exTime{{MainFunc.Name}}[2], _start, _stop);
  
  cudaEventRecord(_start);
  {% for var in MainFunc.FullImpl.InputData %}
  cudaFree((void*){{var.Name}});
  {% endfor %}
  {% for var in MainFunc.FullImpl.OutputData %}
  cudaFree({{var.Name}});
  {% endfor %}
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  float _timeForFree = 0.0f;
  cudaEventElapsedTime(&_timeForFree, _start, _stop);
  m_exTime{{MainFunc.Name}}[3] += _timeForFree;
  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
  {% if not MainFunc.IsVoid %}
  return _resFromGPU;
  {% endif %}
}

{% endfor %}
{% if HasGetTimeFunc %}

void {{MainClassName}}{{MainClassSuffix}}::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  {% for MainFunc in MainFunctions %}
  {% if MainFunc.OverrideMe %}
  if(std::string(a_funcName) == "{{MainFunc.Name}}" || std::string(a_funcName) == "{{MainFunc.Name}}Block")
  {
    a_out[0] = m_exTime{{MainFunc.Name}}[0];
    a_out[1] = m_exTime{{MainFunc.Name}}[1];
    a_out[2] = m_exTime{{MainFunc.Name}}[2];
    a_out[3] = m_exTime{{MainFunc.Name}}[3];
  }
  {% endif %}
  {% endfor %}
}
{% endif %}