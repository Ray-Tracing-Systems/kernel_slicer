/////////////////////////////////////////////////////////////////////
/////////////  Required  Shader Features ////////////////////////////
/////////////////////////////////////////////////////////////////////
{% if GlobalUseInt8 %} 
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
{% endif %}
{% if GlobalUseInt16 %} 
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
{% endif %}
{% if GlobalUseInt64 %} 
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
{% endif %}

/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
## for Incl in Includes  
#include "{{Incl}}"
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
#ifndef uint32_t
#define uint32_t uint
#endif
#define MAXFLOAT 1e37f
#define MINFLOAT 1e37f
#define FLT_MAX 1e37f
#define FLT_MIN -1e37f
#define FLT_EPSILON 1e-6f
#define unmasked
## for Decl in ClassDecls  
{{Decl.Text}}
## endfor

#include "include/{{UBOIncl}}"

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////
bool isfinite(float x) { return !isinf(x); }
float copysign(float mag, float a_sign) { return abs(mag)*sign(a_sign); }


## for LocalFunc in LocalFunctions  
{{LocalFunc}}

## endfor
#define KGEN_FLAG_RETURN            1
#define KGEN_FLAG_BREAK             2
#define KGEN_FLAG_DONT_SET_EXIT     4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8
#define KGEN_REDUCTION_LAST_STEP    16
