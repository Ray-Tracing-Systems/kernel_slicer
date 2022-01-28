#ifndef RAYMARCH_SAMPLE_RAYMARCH_H
#define RAYMARCH_SAMPLE_RAYMARCH_H

#include <cstdint>
#include "LiteMath.h"
#include <vector>

using namespace LiteMath;
class RayMarcher
{
public:
  RayMarcher(uint32_t a_width, uint32_t a_height) : m_width(a_width), m_height(a_height) {}

  void Init(const std::vector<float> &a_densityField, const int3 &a_gridResolution);
  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {}
  void Execute(uint32_t tidX, uint32_t tidY, uint32_t* out_color __attribute__((size("tidX","tidY"))));
  void kernel_InitEyeRay(uint32_t tidX, uint32_t tidY, float4* rayPosAndNear, float4* rayDirAndFar);
  void kernel_RayMarch(uint32_t tidX, uint32_t tidY, const float4* rayPosAndNear, const float4* rayDirAndFar, uint32_t* out_color);
  virtual void ExecuteBlock(uint32_t tidX, uint32_t tidY, uint32_t * out_color, uint32_t a_numPasses = 1);
  float4 GetCamPos() { return m_camPos; }
  float4x4 GetInvProjViewMat() { return m_invProjView; }
protected:
  void InitView();
  float3 RayFunc(float tmin, float tmax, float *alpha, const float4 *ray_pos, const float4 *ray_dir);
  float SampleDensity(float4 pos, bool trilinear);
  float GetDensity(int x, int y, int z);

  uint32_t m_width;
  uint32_t m_height;

  const float4 BACKGROUND_COLOR = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const float4 SCENE_BOX_MIN = make_float4(-3, -3, -3, 0.0f);
  const float4 SCENE_BOX_MAX = make_float4(+3, +3, +3, 0.0f);

  std::vector<float> m_densityField;
  int4 m_gridResolution;

  float4   m_camPos;
  float4x4 m_invProjView;

};


#endif //RAYMARCH_SAMPLE_RAYMARCH_H
