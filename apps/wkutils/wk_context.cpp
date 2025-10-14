#include "wk_context.h"
#include <iostream>
#include <cassert>

// old implementation
//
//void wk_utils::printDeviceInfo(WGPUAdapter adapter) 
//{
//  // Получаем свойства адаптера
//  WGPUAdapterProperties props = {};
//  wgpuAdapterGetProperties(adapter, &props);
//  // Печатаем имя адаптера (драйвер/видеокарта)
//  std::cout << "[wgpu]: Name      : " << props.name << std::endl;
//  if(props.driverDescription != nullptr)
//  std::cout << "[wgpu]: Driver    : " << props.driverDescription << std::endl;
//  std::cout << "[wgpu]: Backend   : " << props.backendType << std::endl;
//  std::cout << "[wgpu]: Device ID : " << props.deviceID << std::endl;
//}

void wk_utils::printDeviceInfo(WGPUAdapter adapter) 
{
  std::cout << "WGPUAdapter name is not avaliable in new WGPU implementations ... " << std::endl;
}

static void onDeviceError(WGPUErrorType type, const char* message, void*) 
{
  std::cout << "[wgpu device error]: " << message << std::endl;
}

//struct UserData 
//{
//  WGPUAdapter adapter = nullptr;
//  bool requestEnded = false;
//};
//
//static void onAdapterRequestEnded(WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* pUserData) 
//{
//  UserData* userData = reinterpret_cast<UserData*>(pUserData);
//  if (status == WGPURequestAdapterStatus_Success) {
//      userData->adapter = adapter;
//  } else {
//      std::cout << "Could not get WebGPU adapter: " << message << std::endl;
//  }
//  userData->requestEnded = true;
//}

//static WGPUAdapter requestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options) 
//{
//  UserData userData;
//  wgpuInstanceRequestAdapter(
//      instance,
//      options,
//      onAdapterRequestEnded,
//      &userData
//  );
//  // Wait until the callback sets requestEnded to true
//  while (!userData.requestEnded) {
//      // You may want to yield or sleep a bit here in real code
//  }
//  assert(userData.adapter != nullptr && "Failed to acquire adapter");
//  return userData.adapter;
//}

#define UNUSED(x) (void)(x)

// Utility function to request an adapter synchronously
static void handle_request_adapter(WGPURequestAdapterStatus status,
                                   WGPUAdapter adapter, WGPUStringView message,
                                   void *userdata1, void *userdata2)
{
  UNUSED(status);
  UNUSED(message);
  UNUSED(userdata2);
  *(WGPUAdapter *)userdata1 = adapter;
}

static void handle_request_device(WGPURequestDeviceStatus status,
                                  WGPUDevice device, WGPUStringView message,
                                  void *userdata1, void *userdata2)
{
  UNUSED(status);
  UNUSED(message);
  UNUSED(userdata2);
  *(WGPUDevice *)userdata1 = device;
}

static WGPUAdapter requestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options) 
{
  WGPUAdapter adapter;
  WGPURequestAdapterOptions adapterOpts = {};
  adapterOpts.nextInChain = nullptr;

  const WGPURequestAdapterCallbackInfo adapterCallbackInfo = {
    .callback = handle_request_adapter,
    .userdata1 = &adapter
  };

  wgpuInstanceRequestAdapter(instance, &adapterOpts, adapterCallbackInfo);
  //std::cout << "WGPU adapter: " << adapter << std::endl;
  return adapter;
}

static WGPUDevice requestDeviceSync(WGPUAdapter adapter, const WGPUDeviceDescriptor* descriptor) 
{
  struct UserData {
      WGPUDevice device = nullptr;
      bool requestEnded = false;
  } userData;
  auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status, WGPUDevice device, const char* message, void* pUserData) {
      UserData* userData = reinterpret_cast<UserData*>(pUserData);
      if (status == WGPURequestDeviceStatus_Success) {
          userData->device = device;
      } else {
          std::cout << "Could not get WebGPU device: " << message << std::endl;
      }
      userData->requestEnded = true;
  };

  WGPUDevice device;

  const WGPURequestDeviceCallbackInfo deviceCallbackInfo = {
    .callback = handle_request_device,
    .userdata1 = &device
  };

  wgpuAdapterRequestDevice(adapter, NULL, deviceCallbackInfo);
  std::cout << "WGPU device: " << device << std::endl;

  //wgpuAdapterRequestDevice(adapter, descriptor, onDeviceRequestEnded, &userData);
  //while (!userData.requestEnded) {
  //    // Optionally sleep/yield here
  //}
  //assert(userData.device != nullptr && "Failed to acquire device");
  //return userData.device;
  return device;
}

wk_utils::WulkanContext wk_utils::globalContextInit(WulkanDeviceFeatures a_features)
{
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

  WGPURequestAdapterOptions adapterOpts = {};
  adapterOpts.nextInChain     = nullptr;
  adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

  res.physicalDevice = requestAdapterSync(res.instance, &adapterOpts);
  printDeviceInfo(res.physicalDevice);

  //  Create device
  WGPUDeviceDescriptor deviceDesc = {};
  deviceDesc.nextInChain = nullptr;
  //deviceDesc.label = "Cur Device"; // Optional: for debugging
  //deviceDesc.requiredFeaturesCount = 0; // No special features
  deviceDesc.requiredFeatures = nullptr;
  deviceDesc.requiredLimits = nullptr;
  deviceDesc.defaultQueue.nextInChain = nullptr;
  //deviceDesc.defaultQueue.label = "The default queue";

  res.device = requestDeviceSync(res.physicalDevice, &deviceDesc);
  //wgpuDeviceSetUncapturedErrorCallback(res.device, onDeviceError, nullptr);

  return res;
}