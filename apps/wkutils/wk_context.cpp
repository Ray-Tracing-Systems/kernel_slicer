#include "wk_context.h"
#include <iostream>
#include <cassert>

#include <thread>
#include <chrono>
#include <mutex>

using wk_utils::WulkanContext;

std::string_view toStdStringView(WGPUStringView wgpuStringView) {
    return
        wgpuStringView.data == nullptr
        ? std::string_view()
        : wgpuStringView.length == WGPU_STRLEN
        ? std::string_view(wgpuStringView.data)
        : std::string_view(wgpuStringView.data, wgpuStringView.length);
}

/**
 * Utility function to get a WebGPU device, so that
 *     WGPUDevice device = requestDeviceSync(adapter, options);
 * is roughly equivalent to
 *     const device = await adapter.requestDevice(descriptor);
 * It is very similar to requestAdapter
 */
WGPUDevice requestDeviceSync(WGPUInstance instance, WGPUAdapter adapter, WGPUDeviceDescriptor const * descriptor) {
    struct UserData {
        WGPUDevice device = nullptr;
        bool requestEnded = false;
    };
    UserData userData;

    // The callback
    auto onDeviceRequestEnded = [](
        WGPURequestDeviceStatus status,
        WGPUDevice device,
        WGPUStringView message,
        void* userdata1,
        void* /* userdata2 */
    ) {
        UserData& userData = *reinterpret_cast<UserData*>(userdata1);
        if (status == WGPURequestDeviceStatus_Success) {
            userData.device = device;
        } else {
            std::cerr << "Error while requesting device: " << toStdStringView(message) << std::endl;
        }
        userData.requestEnded = true;
    };

    // Build the callback info
    WGPURequestDeviceCallbackInfo callbackInfo = {
        /* nextInChain = */ nullptr,
        /* mode = */ WGPUCallbackMode_AllowProcessEvents,
        /* callback = */ onDeviceRequestEnded,
        /* userdata1 = */ &userData,
        /* userdata2 = */ nullptr
    };

    // Call to the WebGPU request adapter procedure
    wgpuAdapterRequestDevice(adapter, descriptor, callbackInfo);

    // Hand the execution to the WebGPU instance until the request ended
    wgpuInstanceProcessEvents(instance);
    while (!userData.requestEnded) {
         std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wgpuInstanceProcessEvents(instance);
    }

    return userData.device;
}

void onAdapterRequestEnded(
    WGPURequestAdapterStatus  status,  // a success status
    WGPUAdapter               adapter, // the returned adapter
    WGPUStringView            message, // optional error message
    void* userdata1,                   // custom user data, as provided when requesting the adapter
    void* userdata2) 
{
  WulkanContext* pContext = (WulkanContext*)userdata1;
  bool* pRequestEnded     = (bool*)(userdata2);
  *pRequestEnded = true;
  pContext->physicalDevice = adapter;
}

wk_utils::WulkanContext wk_utils::globalContextInit(WulkanDeviceFeatures a_features)
{
  //dawnProcSetProcs(&dawn::native::GetProcs());
  WulkanContext res = {};
  
  // The vector size
  WGPUInstanceDescriptor desc = {};
  desc.nextInChain = nullptr;

  // We create the instance using this descriptor
  res.instance = wgpuCreateInstance(&desc);

  if (!res.instance) 
  {
    std::cout << "[wk_utils]: could not initialize WebGPU!" << std::endl;
    return res;
  }

  bool requestEnded = false;

  // Build callback info
  WGPURequestAdapterCallbackInfo callbackInfo = {
    .nextInChain = nullptr,
    .mode        = WGPUCallbackMode_AllowProcessEvents, // more on this later
    .callback    = onAdapterRequestEnded,
    .userdata1   = &res,
    .userdata2   = &requestEnded, // custom user data is simply a pointer to a boolean in this case
  };

  // Start the request
  WGPURequestAdapterOptions options = {};
  options.nextInChain = nullptr;
  wgpuInstanceRequestAdapter(res.instance, &options, callbackInfo);
  wgpuInstanceProcessEvents (res.instance);
  while (!requestEnded) {
    // Hand the execution to the WebGPU instance so that it can check for
    // pending async operations, in which case it invokes our callbacks.
    wgpuInstanceProcessEvents(res.instance);

    // Even if waiting for 10ms before testing again, this is a terrible idea
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  WGPUDeviceDescriptor deviceDesc = WGPU_DEVICE_DESCRIPTOR_INIT;
  res.device = requestDeviceSync(res.instance, res.physicalDevice, &deviceDesc);

  return res;
}