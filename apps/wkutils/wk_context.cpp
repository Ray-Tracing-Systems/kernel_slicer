#include "wk_context.h"
#include <iostream>
#include <cassert>
#include <cstring>

void wk_utils::printDeviceInfo(WGPUAdapter adapter) 
{
  // Получаем свойства адаптера
  WGPUAdapterProperties props = {};
  wgpuAdapterGetProperties(adapter, &props);
  // Печатаем имя адаптера (драйвер/видеокарта)
  std::cout << "[wgpu]: Name      : " << props.name << std::endl;
  if(props.driverDescription != nullptr)
  std::cout << "[wgpu]: Driver    : " << props.driverDescription << std::endl;
  std::cout << "[wgpu]: Backend   : " << props.backendType << std::endl;
  std::cout << "[wgpu]: Device ID : " << props.deviceID << std::endl;
}

static void onDeviceError(WGPUErrorType type, const char* message, void*) 
{
  std::cout << "[wgpu device error]: " << message << std::endl;
}

struct UserData 
{
  WGPUAdapter adapter = nullptr;
  bool requestEnded = false;
};

static void onAdapterRequestEnded(WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* pUserData) 
{
  UserData* userData = reinterpret_cast<UserData*>(pUserData);
  if (status == WGPURequestAdapterStatus_Success) {
      userData->adapter = adapter;
  } else {
      std::cout << "Could not get WebGPU adapter: " << message << std::endl;
  }
  userData->requestEnded = true;
}

static WGPUAdapter requestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options) 
{
  UserData userData;
  wgpuInstanceRequestAdapter(
      instance,
      options,
      onAdapterRequestEnded,
      &userData
  );
  // Wait until the callback sets requestEnded to true
  while (!userData.requestEnded) {
      // You may want to yield or sleep a bit here in real code
  }
  assert(userData.adapter != nullptr && "Failed to acquire adapter");
  return userData.adapter;
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
  wgpuAdapterRequestDevice(adapter, descriptor, onDeviceRequestEnded, &userData);
  while (!userData.requestEnded) {
      // Optionally sleep/yield here
  }
  assert(userData.device != nullptr && "Failed to acquire device");
  return userData.device;
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
  deviceDesc.label = "Cur Device"; // Optional: for debugging
  //deviceDesc.requiredFeaturesCount = 0; // No special features
  deviceDesc.requiredFeatures = nullptr;
  deviceDesc.requiredLimits = nullptr;
  deviceDesc.defaultQueue.nextInChain = nullptr;
  deviceDesc.defaultQueue.label = "The default queue";

  res.device = requestDeviceSync(res.physicalDevice, &deviceDesc);
  wgpuDeviceSetUncapturedErrorCallback(res.device, onDeviceError, nullptr);

  return res;
}

void wk_utils::readBufferBack(WulkanContext a_ctx, WGPUQueue a_queue, WGPUBuffer a_buffer, WGPUBuffer a_tmpBuffer, size_t a_size, void* a_data)
{
  WGPUCommandEncoder encoderRB = wgpuDeviceCreateCommandEncoder(a_ctx.device, nullptr);
  wgpuCommandEncoderCopyBufferToBuffer(encoderRB, a_buffer, 0, a_tmpBuffer, 0, a_size);

  WGPUCommandBuffer cmdRB = wgpuCommandEncoderFinish(encoderRB, nullptr);
  //wgpuCommandEncoderRelease(encoderRB); //removed function ?
  wgpuQueueSubmit(a_queue, 1, &cmdRB);
  
  // 10. Map and read back result
  struct Context {
    bool ready;
    WGPUBuffer buffer;
  };
  Context context = { false, a_tmpBuffer };

  auto onBuffer2Mapped = [](WGPUBufferMapAsyncStatus status, void* pUserData) {
    Context* context = reinterpret_cast<Context*>(pUserData);
    context->ready = true;
    if (status != WGPUBufferMapAsyncStatus_Success) 
    {
      std::cout << "[Mandelbrot_WGPU::ReadBufferBack]: buffer mapped with status " << status << std::endl;
      return;
    }
  };

  wgpuBufferMapAsync(a_tmpBuffer, WGPUMapMode_Read, 0, a_size, onBuffer2Mapped, (void*)&context);

  while (!context.ready) 
  {
    #ifdef USE_DAWN
    wgpuDeviceTick(a_ctx.device);
    #else
    wgpuDevicePoll(a_ctx.device, false, nullptr);
    #endif
  }

  const float* mapped = static_cast<const float*>(wgpuBufferGetConstMappedRange(a_tmpBuffer, 0, a_size));
  std::memcpy(a_data, mapped, a_size);
  wgpuBufferUnmap(a_tmpBuffer);
}