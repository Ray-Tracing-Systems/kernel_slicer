#include <android/log.h>
#include <android_native_app_glue.h>
#include <unordered_map>
#include "volk.h"
#include "render/points_render.h"
#include "android_input.h"

#ifdef NDEBUG
const bool g_enableValidationLayers = false;
#else
const bool g_enableValidationLayers = true;
#endif

extern AppInput g_appInput;

void InitSurface(android_app* app,  IRender* vk_render)
{
  std::vector<const char*> instanceExtensions = {"VK_KHR_surface", VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
  vk_render->InitVulkan(instanceExtensions.data(), instanceExtensions.size(), 0);

  VkAndroidSurfaceCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
  createInfo.pNext = nullptr,
  createInfo.flags = 0,
  createInfo.window = app->window;

  VkSurfaceKHR surface = VK_NULL_HANDLE;
//  __android_log_print(ANDROID_LOG_INFO, "N-Body sample", "vkCreateAndroidSurfaceKHR: 0x%x", vkCreateAndroidSurfaceKHR);

  VK_CHECK_RESULT(vkCreateAndroidSurfaceKHR(vk_render->GetVkInstance(), &createInfo, nullptr, &surface));

  vk_render->InitPresentation(surface);
}

void handle_cmd(android_app* app, int32_t cmd)
{
  assert(app->userData != NULL);
  IRender* vk_base_app = reinterpret_cast<IRender*>(app->userData);
  switch (cmd)
  {
  case APP_CMD_INIT_WINDOW:
    // The window is being shown, get it ready.
    InitSurface(app, vk_base_app);
    vk_base_app->LoadScene("", true);
    break;
  case APP_CMD_TERM_WINDOW:
    // The window is being hidden or closed, clean it up.
    //vk_app->Cleanup();
//    vk_app = nullptr;
    break;
  default:
    __android_log_print(ANDROID_LOG_INFO, "N-Body sample", "event not handled: %d", cmd);
  }
}


void android_main(struct android_app* app) 
{
  std::shared_ptr<IRender> vk_app = CreateRender(1280 * 2, 720 * 2, RenderEngineType::POINTS_RENDER);
  app->userData = vk_app.get();

//  if (!vk_android::InitVulkan()) // load vulkan shared library and functions
//  {
//    LOGW("Vulkan is unavailable");
//    ANativeActivity_finish(app->activity);
//  }
//  else
  {
    AConfiguration* config = AConfiguration_new();
    AConfiguration_fromAssetManager(config, app->activity->assetManager);
    vk_android::screenDensity = AConfiguration_getDensity(config);
    AConfiguration_delete(config);

    app->onAppCmd = handle_cmd;
    app->onInputEvent = vk_base_android_input_handler;
    int events;
    android_poll_source *source;
    do
    {
      if (ALooper_pollAll(vk_app->IsReady() ? 1 : 0, nullptr, &events, (void **) &source) >= 0)
      {
        if (source != NULL)
        {
          source->process(app, source);
          vk_app->ProcessInput(g_appInput);
        }
      }

      if (vk_app->IsReady())
      {
        vk_app->DrawFrame(0.0f);
      }
    } while (app->destroyRequested == 0);
  }
}
