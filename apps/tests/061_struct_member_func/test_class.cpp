#include "test_class.h"

float SphereLight::Attenuation(float a_dist) const
{
  const float r = Radius();
  return r*r/(r*r + a_dist*a_dist);
}

void StructMembers::kernel1D_Shade(float* a_out, uint32_t a_size)
{
  for(uint32_t i=0; i<a_size; i++)
  {
    const float3 point  = float3(float(i % 32), float(i / 32), 0.0f);
    const float3 normal = float3(0.0f, 0.0f, 1.0f);
    float summ = 0.0f;
    for(uint32_t lightId = 0; lightId < uint32_t(m_lights.size()); lightId++)
    {
      SphereLight light = m_lights[lightId];
      light.ScaleTwice(0.75f);
      summ += 0.01f*m_lights[lightId].Intensity();
      summ += light.Eval(point, normal) + m_lights[lightId].Radius();
      SphereLight other(m_lights[lightId].posAndRadius, float4(0.5f, 0.25f, 1.0f, 1.0f));
      summ += other.Attenuation(length(other.Pos() - point))*SphereLight::Luminance(other.color);
    }
    a_out[i] = summ;
  }
}

void StructMembers::Run(float* a_out, uint32_t a_size)
{
  kernel1D_Shade(a_out, a_size);
}
