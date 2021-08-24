#pragma once

bool test000_scalar_funcs();
bool test001_dot_cross_f4();
bool test002_dot_cross_f3();
bool test003_length_float4();
bool test004_colpack_f4x4();
bool test005_matrix_elems();
bool test006_any_all();
bool test007_reflect();
bool test008_normalize();
bool test009_refract();
bool test010_faceforward();
bool test011_mattranspose();

## for Tests in AllTests
## for Test  in Tests.Tests
bool test{{Test.Number}}_basev_{{Test.Type}}();
bool test{{Test.Number+1}}_basek_{{Test.Type}}();
bool test{{Test.Number+2}}_unaryv_{{Test.Type}}();
bool test{{Test.Number+2}}_unaryk_{{Test.Type}}();
bool test{{Test.Number+3}}_cmpv_{{Test.Type}}();
bool test{{Test.Number+4}}_shuffle_{{Test.Type}}();
bool test{{Test.Number+5}}_exsplat_{{Test.Type}}();
bool test{{Test.Number+7}}_funcv_{{Test.Type}}();
{% if Test.IsFloat %}
bool test{{Test.Number+8}}_funcfv_{{Test.Type}}();
bool test{{Test.Number+9}}_cstcnv_{{Test.Type}}();
{% else %}
bool test{{Test.Number+8}}_logicv_{{Test.Type}}();
bool test{{Test.Number+9}}_cstcnv_{{Test.Type}}();
{% endif %}
bool test{{Test.Number+10}}_other_{{Test.Type}}();

## endfor
## endfor