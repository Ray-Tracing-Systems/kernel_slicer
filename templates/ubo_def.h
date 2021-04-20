#ifndef {{MainClassName}}_UBO_H
#define {{MainClassName}}_UBO_H

#include "OpenCLMath.h"

struct {{MainClassName}}_UBO_Data
{
## for Field in UBOStructFields  
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %};
## endfor
  unsigned int dummy_last;
{% for hierarchy in Hierarchies %}
  {% for Impl in hierarchy.Implementations %}
  unsigned int objNum_{{Impl.ClassName}};
  unsigned int objNum_{{Impl.ClassName}}Acc;
  {% endfor %}  
{% endfor %}
};

#endif
