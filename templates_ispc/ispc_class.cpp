#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <cassert>
#include <chrono>

#include "{{MainInclude}}"
#include "{{MainISPCFile}}"
## for Decl in ClassDecls  
{% if Decl.InClass and Decl.IsType %}
using {{Decl.Type}} = {{MainClassName}}::{{Decl.Type}}; // for passing this data type to UBO
{% endif %}
## endfor
#include "LiteMath.h"

struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data
{
  {% for Field in UBO.UBOStructFields %}
  {% if Field.IsDummy %} 
  uint {{Field.Name}}; 
  {% else %}
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %}; 
  {% endif %}
  {% endfor %}
  uint dummy_last;
};

class {{MainClassName}}_ISPC : public {{MainClassName}}
{
public:

  {% for ctorDecl in Constructors %}
  {% if ctorDecl.NumParams == 0 %}
  {{ctorDecl.ClassName}}_ISPC() {}
  {% else %}
  {{ctorDecl.ClassName}}_ISPC({{ctorDecl.Params}}) : {{ctorDecl.ClassName}}({{ctorDecl.PrevCall}}) {}
  {% endif %}
  {% endfor %}
  ~{{MainClassName}}_ISPC(){}

  {% if HasCommitDeviceFunc %}
  void CommitDeviceData() override 
  {
    UpdatePlainMembers();
  }  
  {% endif %}
  {% if HasGetTimeFunc %}
  void GetExecutionTime(const char* a_funcName, float a_out[4]); 
  {% endif %}

  {% for MainFunc in MainFunctions %}
  {{MainFunc.ReturnType}} {{MainFunc.DeclOrig}} override;
  {% endfor %}

protected:

  virtual void UpdatePlainMembers();
  virtual void ReadPlainMembers();

  {% for Kernel in Kernels %}
  void {{Kernel.Name}}({% for Arg in Kernel.Args %}{% if not Arg.IsUBO and not Arg.IsMember %}{% if Arg.IsPointer %}{{Arg.Type}}* {{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %}, {% endif %}{% else %}{{Arg.Type}} {{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %}, {% endif %} {% endif %}{% endif %} {% endfor %}) {% if not Kernel.InitKPass %} override {% endif %}
  { 
    {% if Kernel.OpenMPAndISPC %}
    constexpr int BLOCK_SIZE = 64;
    {% if Kernel.threadDim == 1 %}
    #pragma omp parallel for 
    for(int {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}}; {{Kernel.ThreadId0.Name}} < {{Kernel.ThreadId0.Size}}; {{Kernel.ThreadId0.Name}} += BLOCK_SIZE) {
      const int end = std::min({{Kernel.ThreadId0.Name}} + BLOCK_SIZE, int({{Kernel.ThreadId0.Size}}));
      ispc::{{Kernel.Name}}_ISPC({% for Arg in Kernel.Args %}{% if Arg.IsPointer and Arg.IsMember %}{{Arg.NameISPC}}.data(){% else %}{{Arg.NameISPC}}{%endif%}{% if loop.index1 != Kernel.LastArgNF1 %},{% endif %}{% endfor %},&m_uboData,{{Kernel.ThreadId0.Name}},end,0);  
    }
    {% else %}
    #pragma omp parallel for 
    for(int {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}}; {{Kernel.ThreadId0.Name}} < {{Kernel.ThreadId0.Size}}; {{Kernel.ThreadId0.Name}}++) {
      for(int {{Kernel.ThreadId1.Name}} = {{Kernel.ThreadId1.Start}}; {{Kernel.ThreadId1.Name}} < {{Kernel.ThreadId1.Size}}; {{Kernel.ThreadId1.Name}} += BLOCK_SIZE) {
        const int end = std::min({{Kernel.ThreadId1.Name}} + BLOCK_SIZE, int({{Kernel.ThreadId1.Size}}));
        ispc::{{Kernel.Name}}_ISPC({% for Arg in Kernel.Args %}{% if Arg.IsPointer and Arg.IsMember %}{{Arg.NameISPC}}.data(){% else %}{{Arg.NameISPC}}{%endif%}{% if loop.index1 != Kernel.LastArgNF1 %},{% endif %}{% endfor %},&m_uboData,{{Kernel.ThreadId1.Name}},end,{{Kernel.ThreadId0.Name}});  
      }
    }
    {% endif %}
    {% else %}
    ispc::{{Kernel.Name}}_ISPC({% for Arg in Kernel.Args %}{% if Arg.IsPointer and Arg.IsMember %}{{Arg.NameISPC}}.data(){% else %}{{Arg.NameISPC}}{%endif%}{% if loop.index1 != Kernel.LastArgNF1 %},{% endif %}{% endfor %},&m_uboData); 
    {% endif %}
  }
  {% endfor %}

  ispc::{{MainClassName}}{{MainClassSuffix}}_UBO_Data m_uboData;
  std::unordered_map<std::string, float> m_exTimeISPC;
};

void {{MainClassName}}_ISPC::UpdatePlainMembers()
{
  const size_t maxAllowedSize = std::numeric_limits<uint32_t>::max();
## for Var in ClassVars
  {% if Var.IsArray %}
  memcpy(m_uboData.{{Var.Name}},{{Var.Name}},sizeof({{Var.Name}}));
  {% else %}
  m_uboData.{{Var.Name}} = {{Var.Name}};
  {% endif %}
## endfor
## for Var in ClassVectorVars 
  m_uboData.{{Var.Name}}_size     = uint32_t( {{Var.Name}}{{Var.AccessSymb}}size() );     assert( {{Var.Name}}{{Var.AccessSymb}}size()     < maxAllowedSize );
  m_uboData.{{Var.Name}}_capacity = uint32_t( {{Var.Name}}{{Var.AccessSymb}}capacity() ); assert( {{Var.Name}}{{Var.AccessSymb}}capacity() < maxAllowedSize );
## endfor
}

void {{MainClassName}}_ISPC::ReadPlainMembers()
{
  {% for Var in ClassVars %}
  {% if not Var.IsConst %}
  {% if Var.IsArray %}
  memcpy({{Var.Name}}, m_uboData.{{Var.Name}}, sizeof({{Var.Name}}));
  {% else %}
  {{Var.Name}} = m_uboData.{{Var.Name}};
  {% endif %}
  {% endif %} {#/* end of if not var.IsConst */#}
  {% endfor %}
  {% for Var in ClassVectorVars %}
  {{Var.Name}}{{Var.AccessSymb}}resize(m_uboData.{{Var.Name}}_size);
  {% endfor %}
}

{% for MainFunc in MainFunctions %}
{{MainFunc.ReturnType}} {{MainClassName}}_ISPC::{{MainFunc.DeclOrig}}
{
  auto before = std::chrono::high_resolution_clock::now();
  UpdatePlainMembers();
  {{MainClassName}}::{{MainFunc.Name}}({% for Arg in MainFunc.InOutVars %}{{Arg.Name}}{% if loop.index1 != MainFunc.InOutVarsNum %},{% endif %}{% endfor %});
  ReadPlainMembers();
  m_exTimeISPC["{{MainFunc.Name}}"] = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}
{% endfor %}
{% if HasGetTimeFunc %}

void {{MainClassName}}_ISPC::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  float res = 0.0f;
  {% for MainFunc in MainFunctions %}
  {% if MainFunc.OverrideMe %}
  if(std::string(a_funcName) == "{{MainFunc.Name}}" || std::string(a_funcName) == "{{MainFunc.Name}}Block")
    res = m_exTimeISPC["{{MainFunc.Name}}"];
  {% endif %}
  {% endfor %}
  a_out[0] = res;
  a_out[1] = 0.0f;
  a_out[2] = 0.0f;
  a_out[3] = 0.0f;             
}
{% endif %}

{% for ctorDecl in Constructors %}
{% if ctorDecl.NumParams == 0 %}
std::shared_ptr<{{MainClassName}}> Create{{MainClassName}}_ISPC() 
{ 
  auto pObj = std::make_shared<{{MainClassName}}_ISPC>(); 
  return pObj;
}
{% else %}
std::shared_ptr<{{MainClassName}}> Create{{ctorDecl.ClassName}}_ISPC({{ctorDecl.Params}}) 
{ 
  auto pObj = std::make_shared<{{MainClassName}}_ISPC>({{ctorDecl.PrevCall}}); 
  return pObj;
}
{% endif %}
{% endfor %}