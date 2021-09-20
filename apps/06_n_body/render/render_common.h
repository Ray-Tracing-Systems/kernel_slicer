#ifndef CHIMERA_RENDER_COMMON_H
#define CHIMERA_RENDER_COMMON_H

#include "volk.h"
#include "vk_utils.h"
#include "../Camera.h"
#include <cstring>
#include <memory>

struct AppInput
{
  AppInput(){
    cams[1].pos    = float3(4.0f, 4.0f, 4.0f);
    cams[1].lookAt = float3(0, 0, 0);
    cams[1].up     = float3(0, 1, 0);
  }

  enum {MAXKEYS = 384};
  Camera cams[2];
  bool   keyPressed[MAXKEYS]{};
  bool   keyReleased[MAXKEYS]{};
  void clearKeys() { memset(keyPressed, 0, MAXKEYS*sizeof(bool)); memset(keyReleased, 0, MAXKEYS*sizeof(bool)); }

#if defined(__ANDROID__)
  int32_t touch_pos_x;
  int32_t touch_pos_y;
  bool touchDown = false;
  double touchTimer = 0.0;
  int64_t lastTapTime = 0;
#endif
};

struct pipeline_data_t
{
  VkPipelineLayout layout;
  VkPipeline pipeline;
};


class IRender
{
public:
  virtual uint32_t     GetWidth() const = 0;
  virtual uint32_t     GetHeight() const = 0;
  virtual VkInstance   GetVkInstance() const = 0;

  virtual void InitVulkan(const char** a_instanceExtensions, uint32_t a_instanceExtensionsCount, uint32_t a_deviceId) = 0;
  virtual void InitPresentation(VkSurfaceKHR& a_surface) = 0;
  virtual void ProcessInput(const AppInput& input) = 0;
  virtual void UpdateCamera(const Camera* cams, uint32_t a_camsCount) = 0;
  virtual void LoadScene(const char* path, bool transpose_inst_matrices) = 0;
  virtual void DrawFrame(float a_time) = 0;
  virtual bool IsReady() const = 0;
  virtual ~IRender() = default;

};

enum class RenderEngineType
{
  POINTS_RENDER
};

std::unique_ptr<IRender> CreateRender(uint32_t w, uint32_t h, RenderEngineType type);

#endif//CHIMERA_RENDER_COMMON_H
