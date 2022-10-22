#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <cassert>
#include <chrono>

{{MainInclude}}
#include "{{MainISPCFile}}"
## for Decl in ClassDecls  
{% if Decl.InClass and Decl.IsType %}
using {{Decl.Type}} = {{MainClassName}}::{{Decl.Type}}; // for passing this data type to UBO
{% endif %}
## endfor
#include "include/{{UBOIncl}}"

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
  void {{Kernel.Name}}({% for Arg in Kernel.Args %}{% if not Arg.IsUBO and not Arg.IsMember %}{% if Arg.IsPointer %}{{Arg.Type}}* {{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %}, {% endif %}{% else %}{{Arg.Type}} {{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %}, {% endif %} {% endif %}{% endif %} {% endfor %}) override { ispc::{{Kernel.Name}}_ISPC({% for Arg in Kernel.Args %}{% if Arg.IsPointer and Arg.IsMember %}{{Arg.NameISPC}}.data(){% else %}{{Arg.NameISPC}}{%endif%}{% if loop.index1 != Kernel.LastArgNF1 %},{% endif %}{% endfor %},&m_uboData); }
  {% endfor %}

  ispc::{{MainClassName}}_UBO_Data m_uboData;
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
  m_uboData.{{Var.Name}}_size     = uint32_t( {{Var.Name}}.size() );    assert( {{Var.Name}}.size() < maxAllowedSize );
  m_uboData.{{Var.Name}}_capacity = uint32_t( {{Var.Name}}.capacity() ); assert( {{Var.Name}}.capacity() < maxAllowedSize );
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
  {{Var.Name}}.resize(m_uboData.{{Var.Name}}_size);
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

std::shared_ptr<{{MainClassName}}> Create{{MainClassName}}_ISPC() 
{ 
  auto pObj = std::make_shared<{{MainClassName}}_ISPC>(); 
  return pObj;
}